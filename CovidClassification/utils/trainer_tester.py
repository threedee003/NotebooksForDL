def train(model,train_loader,validation_loader,loss_fn,optimizer,n_epochs,device):
    model = model.to(device)
    for epoch in range(n_epochs):
        training_loss = 0
        train_corr = 0
        totals = 0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input ,target = batch
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            loss = loss_fn(output,target)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()*target.size(0)
            totals += target.size(0)
            predicted = torch.argmax(output,1)
            train_corr += (predicted == target).sum().item()



        validation_loss = 0
        val_corr = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for batch in validation_loader:
                input ,target = batch
                input = input.to(device)
                target = target.to(device)
                output = model(input)
                loss = loss_fn(output,target)
                total += target.size(0)
                validation_loss += loss.item()*target.size(0)
                predicted = torch.argmax(output,1)
                val_corr += (predicted == target).sum().item()


        train_acc = train_corr/totals*100
        val_acc = val_corr/total*100
        training_loss = training_loss/totals
        validation_loss = validation_loss/total
        train_losses.append(training_loss)
        val_losses.append(validation_loss)
        train_accuracy.append(train_acc)
        val_accuracy.append(val_acc)


    
        print('Epoch : {}/{} ...Training Acc : {:.6f}     Training Loss : {:.6f}     Validation Acc : {:.6f}     Validation Loss : {:.6f}'.format(epoch+1,n_epochs,train_acc,training_loss,val_acc,validation_loss))





      













def test(model,test_loader,loss_fn,n_epochs,device=device):
    test_loss = 0
    test_corr = 0
    totals = 0
    y_pred = []
    y_true = []
    model.eval()
    for epoch in range(n_epochs):
        with torch.no_grad():
            for data in test_loader:
                input ,target = data
                input = input.to(device)
                target = target.to(device)
                output = model(input)
                loss = loss_fn(output,target)
                test_loss += loss.item()*target.size(0)
                totals += target.size(0)
                predicted = torch.argmax(output,1)
                test_corr = (predicted==target).sum().item()
                y_true.append(target)
                y_pred.append(predicted)

        test_loss = test_loss/totals
        test_acc = test_corr/totals*100
        test_losses.append(test_loss)
        test_accuracy.append(test_acc)
        print('----------------------------------------------------------------------------')
        print('Y_true:{}'.format(y_true))
        print('----------------------------------------------------------------------------')
        print('Y_pred:{}'.format(y_pred))

    
