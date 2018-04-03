set /p dir="Enter save dir: "

call activate tensorflow
call tensorboard --logdir="D:\PyCharm Projects\word-rnn-tensorflow\\%dir%"
127.0.0.1:6006

pause

pause

