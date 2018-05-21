SIZE = 15218520
ratio = (60, 20, 20)
BUFFER_SIZE =  1000

train_size = SIZE * ratio[0]//100//BUFFER_SIZE
validation_size = SIZE * ratio[1]//100//BUFFER_SIZE
test_size = SIZE * ratio[2]//100//BUFFER_SIZE

with open("full.txt", 'r', encoding='utf-8') as dataset:
    with open("trian.txt", 'w', encoding='utf-8') as train:
        for i in range(train_size):
            read = dataset.read(BUFFER_SIZE)
            train.write(read)
    with open("validate.txt", 'w', encoding='utf-8') as validate:
        for i in range(validation_size):
            read = dataset.read(BUFFER_SIZE)
            validate.write(read)
    with open("test.txt", 'w', encoding='utf-8') as test:
        for i in range(test_size):
            read = dataset.read(BUFFER_SIZE)
            test.write(read)
        
        
