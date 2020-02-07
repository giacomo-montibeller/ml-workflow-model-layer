# ML Workflow

## Module Layer

In this layer the data produced by the data layer is used to train a machine learning model (in this example a polynomial regression).

The model produced is then versioned with DVC, so it is ready to be used by the serving application.