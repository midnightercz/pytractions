import streamlit as st

st.title("Pytractions fundamentals")
st.markdown(
    """
Fundamentals
------------

Pytractions is python frameworks aimed to simplify the processs of creating complex data pipelines
based on reusable basic blocks connected with each other by IOs. Single block of the pipeline is
called Traction.

Traction constist of 4 public type of arguments: `inputs`, `outputs`, `resources` and `args`

Traction input can be connected to other traction output. Like following (pseudocode):
```pseudocode
Model:
   name: str
   age: int

T1:
    output1: Out[Model]

T2:
    input1: In[Model]

T2.input1 = T1.output1
```
One input can be connected to one output. However one output can be connected to multiple inputs.
Input and output classes are containers for atomic data.
Input attribute has to be connected only to  output (or input) attribute of the same type.
It's not possible to connect input to inner structure defined within an output.
As defined above, it's not possible to do (pseudocode):
```
T3
  input: In[str]

T3.input = T1.output1.name
```

### Core ideas

* Traction constist of 4 public type of arguments: `inputs`, `outputs`, `resources` and `args`
* Traction consists of public attribute definitions and definion of `run` method which processes
inputs and produces outputs. Anything else (if needed) should be declared as private.
* Traction `run` method is supposed to process input data and produce output data.
Obtaining, storing or updating data to external resources should be done using `resources`.
* All used external functionality is defined in `resources` and passed to traction when it's
initialized to make testing easier.
* Traction can have zero or more inputs and zero or more outputs. It's prefered to have more
outputs than
one complex output. More outputs makes traction easier to connect to other tractions.
As example, traction FoodSorter is better defined with outputs: `meat`, `vegetables`, `fruits`
rather than with output `sorted_food: {meat: [], vegetables: [], fruits: []}`

* After traction process is finished, all information about it's execution are available in
traction instance attributes.
* Everything publically available within a traction is json compatible
  [typing-system](#typing-system).

Typing system
-------------
Tractions are supposed to work with non-binary data which are consisting of json compatible
elementary types. These types are: `bool`, `string`, `int`, `float`, `list` and `dict`.
Supported is also any
combination of these types and any json compatible class consisted of these types.
More information about typing is available here [typing](developer_guide#pytraction-base).

Traction attributes
-------------------

**Inputs and outputs**

Inputs and ouputs are supposed to be json compatible types of models consistent of these which
holds information about the data which are supposed to be processed by traction. Input or output
shouldn't be a class providing methods for data manipulation - those should be rather handled
by traction itself or by a resource.
As an example of wrong input data consider `MyHttpClient` which holds information about
http session configuration and provide also functionality to fetch http pages.
Session should be rather defined as resource which should be

1. Fully initialized as a resource; inputs should be requested urls. Pseudo code below:
```
mhc = MyHttpClient(auth=(...), retries=(...), timeout=(...))

traction run method:
    for input_url in inputs:
        mhc.fetch(input_url)
```

2. Partially initiated as a resource and final configuration should be done per traction
run via traction arguments; inputs should be requested urls.
```
mhc = MyHttpClient(auth=(...))

traction run method:
    retries = args.retries
    timeout = args.timeout
    for input_url in inputs:
        mhc.fetch(input_url, retries=retries, timeout=timeout)
```

3. Partially initialized and final configuration should be passed as input model which
consists of request attributes and url;
```
mhc = MyHttpClient()

traction run method:
    retries = args.retries
    timeout = args.timeout
    for request_data in inputs:
        mhc.fetch(request.data_url,
                  auth=request_data.auth,
                  retries=request_data.retries,
                  timeout=request_data.timeout)
```

Arguments
---------
Any information which is is constant for whole traction process should be defined as argument.
Those can be for example: number of retries, timeout, threshold, number of threads etc

Resources
---------
Resources ment to provide access to external funcionality to generate, update or store data.
Resource class should be written to support elementary operations for the data.
Any complex operations on the
data should be handled by the traction itself.
Example use cases for a resource can be:
- Fetch client data form a database
- Create-Or-Update client data in database
- Delete client data from database

Other use cases should be defined in tractions.
As example of traction we can define: `RemoveInactiveClients`
which should do following:
- Fetch all clients from database
- Filter out inactive clients
- Filter clients scheduled for removal
- For all clients scheduled for ramoval
    - Remove client data from database (via the client resource)
- For all remaning inactive clients
    - Send notifications to inactive clients (via a resource)
    - Set client data to scheduled for removal
    - Udate client data in database (via the client resource)

Joining tractions together
--------------------------
Tration can be connected to each other by t2.o_input1 = t1.i_input.
But to makes things reusable there's
special tration class called `Tractor`. Idea of tractor is simple: define all tractions and their
connections in one class which can be later used as regular traction. Schema is following:
```
Tractor
    i_input: In[int] = In[int](data=0)
    a_arg: Arg[int] = Arg[int](data=0)
    t_traction1: Traction1 = Traction1(uid='1', i_input=i_input, a_arg=a_arg)
    t_traction2: Traction2 = Traction2(uid='2', i_input=t_traction1.o_output, a_arg=a_arg)
    o_output: Out[int] = t_traction2.o_output
```
When tractor is initialized, all tractions are copied from class variables into instance variables
located in `Tractor.traction` list.
You can learn details about tractors in [tractor](developer_guide#tractors) section.
"""
)
