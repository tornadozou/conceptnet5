FROM rspeer/conceptnet-base:5.3b2
MAINTAINER Rob Speer <rob@luminoso.com>

# Configure the environment where ConceptNet will be built
ENV PYTHON python3
ADD conceptnet5 /src/conceptnet/conceptnet5
ADD tests /src/conceptnet/tests
ADD setup.py /src/conceptnet/setup.py
ADD Makefile /src/conceptnet/Makefile

# Set up ConceptNet
WORKDIR /src/conceptnet
RUN python3 setup.py develop
RUN pip3 install assoc_space==1.0.0

# Download 1 GB of input data
RUN make -e download

# Build ConceptNet. This takes between 12 and 36 hours for me, depending
# on the computer I run it on.
#
# -j8 means to use 8 parallel processes.
RUN make -e -j8 all
RUN make -e -j4 build_assoc
