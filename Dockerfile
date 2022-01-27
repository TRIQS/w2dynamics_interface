# See ../triqs/packaging for other options
FROM flatironinstitute/triqs:unstable-ubuntu-clang
ARG APPNAME=w2dynamics_interface

RUN apt-get install -y libnfft3-dev || yum install -y nfft-devel || dnf install -y 'https://download-ib01.fedoraproject.org/pub/epel/7/x86_64/Packages/n/nfft-3.3.2-1.el7.x86_64.rpm' 'https://download-ib01.fedoraproject.org/pub/epel/7/x86_64/Packages/n/nfft-devel-3.3.2-1.el7.x86_64.rpm'

COPY requirements.txt /src/$APPNAME/requirements.txt
RUN pip3 install -r /src/$APPNAME/requirements.txt

COPY --chown=build . $SRC/$APPNAME
WORKDIR $BUILD/$APPNAME
RUN chown build .
USER build
ARG BUILD_DOC=0
ARG BUILD_ID
RUN cmake $SRC/$APPNAME -DTRIQS_ROOT=${INSTALL} -DBuild_Documentation=${BUILD_DOC} -DBuild_Deps=Always && make -j1 || make -j1 VERBOSE=1
USER root
RUN make install
