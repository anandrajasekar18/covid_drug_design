## Run the main file
### With Psuedonomas/train.csv
* Edge Convolutions
`python -W ignore main.py --net_type=Edge data_type=psuedo --num_epochs=51`
* Normal Convolutions
`python -W ignore main.py --net_type=Conv data_type=psuedo --num_epochs=51`

### With 10-fold splits given by MI
* Edge Convolutions
`python -W ignore main.py --net_type=Edge data_type=fold --num_epochs=51`
* Normal Convolutions
`python -W ignore main.py --net_type=Conv data_type=fold --num_epochs=51`
