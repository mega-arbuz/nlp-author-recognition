# Author identification in online theme-based communities
Identifying an author of a given anonymous subreddit message using machine learning and NLP techniques.

## Prerequisites

### Python Version
The python version that is used in the project is 3.6.

### Project Requirements
The python libraries are listed in `requirements.txt` file. To install them, run:
```bash
pip install -r requirements.txt
```
You can also open the project with PyCharm in a new virtual environment and let the IDE install the necessary dependencies.


### Java
The project is using StanfordCoreNLP engine. On the first run, it will download the engine with English resources.
The NLP engine needs Java to be installed and provided inside the `$PATH` variable. 
If there is no Java in `$PATH` or the project runs in a virtual environment, go to `go.py`, uncomment the code on line #34 and modify it as follows:
```python
os.environ['PATH'] += ':/path/to/your/java/bin'
```

## Resources

### NLP Server
This step is not mandatory but can be performed in order to increase the performance. It can also increase accuracy due to low default timeout.

The StanfordCoreNLP is resourceful and thus the project will catch any exception and close the process when the analyzer is terminated.
After the first run, it is possible to avoid starting the NLP server on each run. Execute the following command to run a persistent server:
```bash
./run_stanford_nlp.sh
```
Now there is an NLP server running and when the project is executed with `-na` flag, it will connect to the running server.

### Caching
There is a caching mechanism that saves all the results from the NLP engine in a JSON file.
Caching is enabled by default and when the analyzer is executed for the second time on the same data, it will get the parsing trees from the cache.
The execution time will be ~30 times faster.

## How does it work

### Data
The data which this project includes was collected from different subdreddits, using reddit API and it is stored in the [data folder](https://github.com/mega-arbuz/nlp-author-recognition/tree/master/data). Each data file contains between 120-500 messages from 10 different authors.

### Method
The project contains an implementation of different lexical and syntactic features.
All messages are converted to a feature vector and a Logistic Regression models is trained based on the features.
In each run, a random 80% of messages are used for training and the remaining 20% for test.

### Execution
Help:
```bash
python go.py -h
```
Arguments:
```
usage: go.py [-h] [-na] [-nc] [-s] [-d {movies_120,learn_python_500,dnd_500}]
             [-i ITERATIONS] [-umin USERS_MIN] [-umax USERS_MAX]
             [-f {all,combined,singles,groups,lexical,syntactic,constituency,pos_tags,dependency,sentence_length,message_length,unigram,trigram}]

optional arguments:
  -h, --help            show this help message and exit
  -na                   connect to a running nlp engine instead of starting a
                        new server
  -nc                   don't cache results from nlp engine
  -s                    non verbose print (silent)
  -d {movies_120,learn_python_500,dnd_500}
                        dataset name
  -i ITERATIONS         number of iterations (per feature)
  -umin USERS_MIN       minimum number of users
  -umax USERS_MAX       maximum number of users
  -f {all,combined,singles,groups,lexical,syntactic,constituency,pos_tags,dependency,sentence_length,message_length,unigram,trigram}
                        feature set
```

#### Examples
Simple run with default arguments (10 users, movies data, best features, auto start nlp server, cache enabled, verbose output, one iteration per run)
```bash
python go.py
```
Custom run - run for 2 to 10 users, dnd data, all features, connect to existing nlp server, cache enabled, non verbose output, 3 iterations per run.
The result will be a CSV data with scores for all executions.
```bash
python go.py -f all -na -s -i 3 -umin 2 -umax 10 -d dnd_500
```

## Troubleshooting

### No Java
In case of Java related exception, see instructions above on how to add Java to `$PATH`.

### Mac OS
This project works best on Linux distributions.
If you are executing this project on Mac OS, you might get the following security exception:
```
...
PermissionError: [Errno 1] Operation not permitted

During handling of the above exception, another exception occurred:
...
   raise AccessDenied(self.pid, self._name)
psutil.AccessDenied: psutil.AccessDenied (pid=...)
```
It happens because of a bug in the StanfordCoreNLP library and can be fixed by running a persistent NLP server (instructions above) or by editing the `/path/to/python/packages/stanfordcorenlp/corenlp.py`  as follows:

Comment out or remove the following lines (`corenlp.py:76-85`)
```python
# If port not set, auto select
if self.port is None:
    for port_candidate in range(9000, 65535):
        if port_candidate not in [conn.laddr[1] for conn in psutil.net_connections()]:
            self.port = port_candidate
            break

# Check if the port is in use
if self.port in [conn.laddr[1] for conn in psutil.net_connections()]:
    raise IOError('Port ' + str(self.port) + ' is already in use.')
```

And replace with:

```python
self.port = 9999
```
