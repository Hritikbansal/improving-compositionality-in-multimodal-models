import time
import wandb
import torch
import logging
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast

def reorder(embeds, options):
    num_devices = options.num_devices
    per_gpu_batch_size = options.batch_size
    assert per_gpu_batch_size == len(embeds) // num_devices // 2
    original_embeds = []
    negative_embeds = []
    for i in range(0, len(embeds), per_gpu_batch_size):
        if (i // per_gpu_batch_size)%2 == 0:
            original_embeds.append(embeds[i: i + per_gpu_batch_size])
        else:
            negative_embeds.append(embeds[i: i + per_gpu_batch_size])
    original_embeds = torch.cat(original_embeds)
    negative_embeds = torch.cat(negative_embeds)
    return torch.cat([original_embeds, negative_embeds])

def get_loss(umodel, outputs, criterion, options):  
    if(options.inmodal):
        image_embeds, augmented_image_embeds = outputs.image_embeds[:len(outputs.image_embeds) // 2], outputs.image_embeds[len(outputs.image_embeds) // 2:]
        text_embeds, augmented_text_embeds = outputs.text_embeds[:len(outputs.text_embeds) // 2], outputs.text_embeds[len(outputs.text_embeds) // 2:]
    else:
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
            
    if(options.distributed):
        if(options.inmodal):
            gathered_image_embeds = [torch.zeros_like(image_embeds) for _ in range(options.num_devices)]
            gathered_text_embeds = [torch.zeros_like(text_embeds) for _ in range(options.num_devices)]
            augmented_gathered_image_embeds = [torch.zeros_like(augmented_image_embeds) for _ in range(options.num_devices)]
            augmented_gathered_text_embeds = [torch.zeros_like(augmented_text_embeds) for _ in range(options.num_devices)]
            
            dist.all_gather(gathered_image_embeds, image_embeds)
            dist.all_gather(gathered_text_embeds, text_embeds)
            dist.all_gather(augmented_gathered_image_embeds, augmented_image_embeds)
            dist.all_gather(augmented_gathered_text_embeds, augmented_text_embeds)
            
            image_embeds = torch.cat(gathered_image_embeds[:options.rank] + [image_embeds] + gathered_image_embeds[options.rank + 1:])
            text_embeds  = torch.cat(gathered_text_embeds[:options.rank]+ [text_embeds] + gathered_text_embeds[options.rank + 1:])
            augmented_image_embeds = torch.cat(augmented_gathered_image_embeds[:options.rank] + [augmented_image_embeds] + augmented_gathered_image_embeds[options.rank + 1:])
            augmented_text_embeds  = torch.cat(augmented_gathered_text_embeds[:options.rank]+ [augmented_text_embeds] + augmented_gathered_text_embeds[options.rank + 1:])      
        else:
            gathered_image_embeds = [torch.zeros_like(image_embeds) for _ in range(options.num_devices)]
            gathered_text_embeds = [torch.zeros_like(text_embeds) for _ in range(options.num_devices)]
        
            dist.all_gather(gathered_image_embeds, image_embeds)
            dist.all_gather(gathered_text_embeds, text_embeds)
        
            image_embeds = torch.cat(gathered_image_embeds[:options.rank] + [image_embeds] + gathered_image_embeds[options.rank + 1:])
            text_embeds  = torch.cat(gathered_text_embeds[:options.rank]+ [text_embeds] + gathered_text_embeds[options.rank + 1:])
            if (options.neg_caption_key):
                text_embeds = reorder(text_embeds, options)
            if (options.neg_image_key or options.shuffle_image_patches):
                image_embeds = reorder(image_embeds, options)
        
    logits_text_per_image = umodel.logit_scale.exp() * image_embeds @ text_embeds.t()
    logits_image_per_text = logits_text_per_image.t()

    if options.neg_caption_key:
        logits_image_per_text = logits_image_per_text[:len(logits_image_per_text) // 2]  
    
    if (options.neg_image_key or options.shuffle_image_patches):
        logits_text_per_image = logits_text_per_image[:len(logits_text_per_image) // 2]

    if(options.inmodal):
        logits_image_per_augmented_image = umodel.logit_scale.exp() * image_embeds @ augmented_image_embeds.t()
        logits_text_per_augmented_text = umodel.logit_scale.exp() * text_embeds @ augmented_text_embeds.t()
    
    batch_size = len(logits_text_per_image)    

    contrastive_loss = torch.tensor(0).to(options.device)
    target = torch.arange(batch_size).long().to(options.device, non_blocking = True)

    if(options.inmodal):
        crossmodal_contrastive_loss = (criterion(logits_text_per_image, target) + criterion(logits_image_per_text, target)) / 2
        inmodal_contrastive_loss = (criterion(logits_image_per_augmented_image, target) + criterion(logits_text_per_augmented_text, target)) / 2
        contrastive_loss = (crossmodal_contrastive_loss + inmodal_contrastive_loss) / 2
    else:
        crossmodal_contrastive_loss = (criterion(logits_text_per_image, target) + criterion(logits_image_per_text, target)) / 2
        contrastive_loss = crossmodal_contrastive_loss

    inmodal_cyclic_loss = torch.tensor(0).to(options.device)
    crossmodal_cyclic_loss = torch.tensor(0).to(options.device)

    if not options.neg_caption_key and not options.neg_image_key and not options.shuffle_image_patches:
        if(options.cylambda1 > 0):
            logits_image_per_image = umodel.logit_scale.exp() * image_embeds @ image_embeds.t()
            logits_text_per_text = umodel.logit_scale.exp() * text_embeds @ text_embeds.t()
            inmodal_cyclic_loss = (logits_image_per_image - logits_text_per_text).square().mean() / (umodel.logit_scale.exp() * umodel.logit_scale.exp()) * batch_size
        
        if(options.cylambda2 > 0):
            crossmodal_cyclic_loss = (logits_text_per_image - logits_image_per_text).square().mean() / (umodel.logit_scale.exp() * umodel.logit_scale.exp()) * batch_size

    cyclic_loss = options.cylambda1 * inmodal_cyclic_loss + options.cylambda2 * crossmodal_cyclic_loss
    loss = contrastive_loss + cyclic_loss
    
    return loss, contrastive_loss, cyclic_loss

def train(epoch, model, data, optimizer, scheduler, scaler, options):    
    dataloader = data["train"]
    if(options.distributed): dataloader.sampler.set_epoch(epoch)

    model.train()
    criterion = nn.CrossEntropyLoss().to(options.device)

    modulo = max(1, int(dataloader.num_samples / options.batch_size / 10))
    umodel = model.module if(options.distributed) else model

    start = time.time()
    
    logging.info(f"Num samples: {dataloader.num_samples}, Num_batches: {dataloader.num_batches}")
    for index, batch in enumerate(dataloader): 
        step = dataloader.num_batches * epoch + index
        scheduler(step)

        optimizer.zero_grad()
        
        if(options.inmodal):
            input_ids, attention_mask, pixel_values = batch["input_ids"][0].to(options.device, non_blocking = True), batch["attention_mask"][0].to(options.device, non_blocking = True), batch["pixel_values"][0].to(options.device, non_blocking = True)
            augmented_input_ids, augmented_attention_mask, augmented_pixel_values = batch["input_ids"][1].to(options.device, non_blocking = True), batch["attention_mask"][1].to(options.device, non_blocking = True), batch["pixel_values"][1].to(options.device, non_blocking = True)
            input_ids = torch.cat([input_ids, augmented_input_ids])
            attention_mask = torch.cat([attention_mask, augmented_attention_mask])
            pixel_values = torch.cat([pixel_values, augmented_pixel_values])
        else:
            input_ids, attention_mask, pixel_values = batch["input_ids"].to(options.device, non_blocking = True), batch["attention_mask"].to(options.device, non_blocking = True), batch["pixel_values"].to(options.device, non_blocking = True)

        if(options.neg_caption_key):
            neg_input_ids, neg_attention_mask = batch["negative_input_ids"].to(options.device, non_blocking = True), batch["negative_attention_mask"].to(options.device, non_blocking = True)
            input_ids = torch.cat([input_ids, neg_input_ids])
            attention_mask = torch.cat([attention_mask, neg_attention_mask])
        if(options.neg_image_key or options.shuffle_image_patches):
            neg_pixel_values = batch["negative_pixel_values"].to(options.device, non_blocking = True)
            pixel_values = torch.cat([pixel_values, neg_pixel_values])

        outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)

        with autocast():
            loss, contrastive_loss, cyclic_loss = get_loss(umodel, outputs, criterion, options)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
        
        scaler.update()
        umodel.logit_scale.data = torch.clamp(umodel.logit_scale.data, 0, 4.6052)

        end = time.time()

        if(options.master and (((index + 1) % modulo == 0) or (index == dataloader.num_batches - 1))):
            num_samples = (index + 1) * len(input_ids) * options.num_devices
            dataloader_num_samples = dataloader.num_samples

            logging.info(f"Train Epoch: {epoch:02d} [{num_samples}/{dataloader_num_samples} ({100.0 * (index + 1) / dataloader.num_batches:.0f}%)]\tLoss: {loss.item():.6f}\tTime taken {end - start:.3f}\tLearning Rate: {optimizer.param_groups[0]['lr']:.9f}")

            metrics = {"loss": loss.item(), "contrastive_loss": contrastive_loss.item(), "cyclic_loss": cyclic_loss.item(), "time": end - start, "lr": optimizer.param_groups[0]["lr"]}
            if(options.wandb):
                for key, value in metrics.items():
                    wandb.log({f"train/{key}": value, "step": step})
        
            start = time.time()
