# neural

Forward-feeding neural network (FFNN) in Golang. A practical approach involves using Python for model development and training due to its rich ML ecosystem, then exporting the trained model for deployment in Go. This hybrid solution combines Go's performance and concurrency benefits with Python's strengths in model development.

Train the model in Python.
```
cd neural
cd python
python main.py
```


Run predictions in Go.
```
cd ../golang
go run .
```
