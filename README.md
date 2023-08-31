# Tapex & Tapas & Llama2 Demo

This application tries to draw conclusions using 3 different models 
* Tapex
* Tapas
* LLama2
  
from the tables we obtained using Postgre SQL.

## Getting Started

Follow these steps to get the application up and running on your local machine:

### Prerequisites

Download the repository or clone it:
```
https://github.com/denizataes/Tapex---Tapas---Llama2-Demo.git
```

In order to execute this project, python3 is needed.
```
sudo apt install python3
```

Also, some libraries must be installed:

```
pip install transformers pandas streamlit torch langchain psycopg2
```

## Usage
1- Run the Streamlit application:
```
streamlit run main.py
```
2- The web application will open in your default web browser.

3- Choose the model you will use ( If you are going to use llama2, you have to download the model via huggingface.)

4- Ask your question.


## Contributing
1- Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

2- Fork this repository.

3- Create a new branch with a descriptive name for your feature or bug fix.

4- Make your changes and commit them.

5- Push your changes to your forked repository.

6- Create a pull request here, detailing the changes you've made.
