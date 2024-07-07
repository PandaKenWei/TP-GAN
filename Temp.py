for epoch in range(last_epoch + 1, config.train['num_epochs']):
    best_val_acc = 0 
    best_model = None
    lr_scheduler.step()
    for step, batch in enumerate(train_dataloader):
        # warm up learning rate
        # if config.train['resume_optimizer'] is None and epoch == last_epoch + 1:
        #     optimizer.param_groups[0]['lr'] = lr_warmup(step + 1, config.train['warmup_length']) * config.train['learning_rate']

        for k in batch:
            batch[k] = batch[k].cuda(non_blocking=True)

        set_requires_grad(stem, True)
        predicts, features = stem(batch['img'], use_dropout=True)
        train_acc, train_loss = compute_loss(predicts, batch['label'])

        train_acc_log_list.append(train_acc.item())
        train_loss_log_list.append(train_loss.item())
        train_loss_epoch_list.append(train_loss.item())

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        tb.add_scalar('loss', train_loss.item(), epoch * len(train_dataloader) + step, 'train')
        tb.add_scalar('acc', train_acc.item(), epoch * len(train_dataloader) + step, 'train')
        tb.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch * len(train_dataloader), 'train')

        if step % config.train['log_step'] == 0:
            set_requires_grad(stem, False)
            tt = time.time()
            if not config.test_time:
                acc_num_list, loss_list = [], []
                for idx, val_batch in enumerate(val_dataloader):
                    for k in val_batch:
                        val_batch[k] = val_batch[k].cuda(non_blocking=True)
                    predicts, features = stem(val_batch['img'], use_dropout=False)
                    val_acc, val_loss = compute_loss(predicts, val_batch['label'])
                    val_acc_num = val_acc * predicts.shape[0]

                    loss_list.append(val_loss)
                    acc_num_list.append(val_acc_num)
                val_loss = torch.mean(torch.stack(loss_list))
                val_acc = torch.sum(torch.stack(acc_num_list)) / len(val_dataloader.dataset)

                train_loss = np.mean(train_loss_log_list)
                train_acc = np.mean(train_acc_log_list)

                train_loss_log_list, train_acc_log_list = [], []

                tb.add_scalar('loss', val_loss.item(), epoch * len(train_dataloader) + step, 'val')
                tb.add_scalar('acc', val_acc.item(), epoch * len(train_dataloader) + step, 'val')

                # if best_val_acc < val_acc:
                #     best_val_acc = val_acc
                #     best_model = copy.copy(stem)

                log_msg = ("epoch {} , step {} / {} , train_loss {:.5f}, train_acc {:.2%} , "
                           "val_loss {:.5f} , val_acc {:.2%} {:.1f} imgs/s").format(
                    epoch, step, len(train_dataloader) - 1, train_loss, train_acc, val_loss.item(), 
                    val_acc.item(), config.train['log_step'] * config.train['batch_size'] / (tt - t))
                print(log_msg)
                log_file.write(log_msg + '\n')

            else:
                print("epoch {} , step {} / step {} , data {:.3f}s , mv_to_cuda {:.3f}s "
                      "forward {:.3f}s acc {:.3f}s loss {:.3f}s , backward {:.3f}s".format(
                    epoch, step, len(train_dataloader), t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5))
            t = tt
    # optimizer.param_groups[0]['lr'] *= config.train['learning_rate_decay']
    temp_train_loss = np.mean(train_loss_epoch_list)

    train_loss_epoch_list = []
    train_loss_log_list = []
    train_acc_log_list = []
    # if config.train['auto_adjust_lr']:
    #     auto_adjust_lr(optimizer, pre_train_loss, temp_train_loss)
    # pre_train_loss = temp_train_loss

    save_model(stem, tb.path, epoch)
    save_optimizer(optimizer, stem, tb.path, epoch)
    print("Save done in {}".format(tb.path))
