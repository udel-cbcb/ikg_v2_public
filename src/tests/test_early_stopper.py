from model.utils import EarlyStopper

def test_early_stopper_loss_decreasing_normally():
    early_stopper = EarlyStopper(early_stopping_delta=1,patience=10)

    loss = 100
    stopped_at=0
    for i in range(100):
        
        if i > 30:
            loss = loss - 0.5
        else:
            loss = loss - 1
        early_stopper.update(loss,i)
        print(f"Loss: {loss} / Best loss: {early_stopper.best_loss} / Change delta : {early_stopper.change_delta} / Num epochs waited: {early_stopper.num_of_epochs_waited}")
        if early_stopper.get_should_stop() == True:
            break
        stopped_at = i

    assert stopped_at == 41


def test_early_stopper_loss_increase_and_decreases():
    early_stopper = EarlyStopper(early_stopping_delta=1,patience=10)

    loss = 100
    stopped_at=0
    for i in range(100):
        
        if i > 30 and i < 39:
            loss = loss - 0.5
        elif i >= 39 and i <= 59:
            loss = loss - 1
        elif i > 60:
            loss = loss - 0.5
        else:
            loss = loss - 1
        early_stopper.update(loss,i)
        print(f"Index: {i} / Loss: {loss} / Best loss: {early_stopper.best_loss} / Change delta : {early_stopper.change_delta} / Num epochs waited: {early_stopper.num_of_epochs_waited}")
        if early_stopper.get_should_stop() == True:
            break
        stopped_at = i    
        
    assert stopped_at == 71
