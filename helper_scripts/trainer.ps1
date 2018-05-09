cd "c:\amharic_model\"
gpu;
python "C:\amharic_model\train_char_pred.py" -b 100 -e .10 -s 100 -i 'data/data_100_10000.h5' -t batched
python "C:\amharic_model\train_class_pred.py" -b 100 -e .10 -s 100 -i 'data/data_100_10000.h5' -t batched
Read-Host -Prompt 'Close'

