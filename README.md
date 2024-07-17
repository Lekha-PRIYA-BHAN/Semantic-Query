# Repository: query

Here is the github repo that will query data from an index created using the vector database like FAISS. The index will be created using the repository *ingest*. Thus you should clone "ingest" and execute it to create the index first. The documents in the index will be chunkified documents whose embeddings have been computed using OpenAI.

# Prerequisites

**Python Version**:

Ensure that the version of python should be >= 3.11.0

Run the following command to check what version you have.

```
python --version
```

**OpenAI key**:

Obtain an OpenAI key

**Index has been created**:

Clone the sister repository named "ingest". Execute the instructions in the readme file of that repo to create an index, say `index1`.


# Get ready to query documents from the Vector Database

Open a command prompt on Windows 10 (commands on Linux will be similar :-)

Run the following commands in the following order:

* `1. setup-env.bat`
* `2. prepare-env.bat`

If the above commands are executed without problems then you are ready to query from the index you would have created as mentioned in the Prerequisites.

To be able to query the index do the following:

**Step 1**: Create a file named "`.env`" in the root folder, and put your OpenAI key as the value of the parameter `OPENAI_API_KEY`

for example:

`OPENAI_API_KEY=sk-aQqHVSapwl9J4bz5O2p3T3Blbk4JI3AUNeUOQVPOM5Uv2Tlr`

Step 2: Edit the file chatbot v2.py and set the value of the variable to the filepath of the index1 which you would have created as mentioned in the Prequisites.

for example:

`index_to_process="../ingest/index1"`

**Step 3**: within the python virtual environment created earlier execute the following command:
`streamlit run "chatbot v2.py"`

This should automatically open in your browser the chatbot that would allow you to run the queries.

Try running "chatbot v3.py". This has a sidebar parameter that you can set for specially searching for terraform code. Bear in mind that you have to ingest all the code and architecture documents using the 'ingest' code.
