# See ../triqs/packaging for other options
FROM flatironinstitute/triqs:master-ubuntu-clang

RUN if ! apt-get install -y libnfft3-dev python-configobj ; then \
    yum install -y nfft-devel python-configobj python-pip && \
    yum erase -y numpy && \
    pip install -U numpy scipy ; \
    pip install -U --no-binary=h5py h5py ; \
  fi

ARG APPNAME=w2dynamics_interface
COPY . $SRC/$APPNAME
WORKDIR $BUILD/$APPNAME
RUN chown build .
USER build
ARG BUILD_DOC=0
RUN cmake $SRC/$APPNAME -DTRIQS_ROOT=${INSTALL} -DBuild_Documentation=${BUILD_DOC} && make -j1 && make test CTEST_OUTPUT_ON_FAILURE=1
USER root
RUN make install
