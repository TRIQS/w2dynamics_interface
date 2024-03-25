# See ../triqs/packaging for other options
FROM flatironinstitute/triqs:unstable-ubuntu-clang
ARG APPNAME=w2dynamics_interface

RUN apt-get install -y libnfft3-dev python3-h5py python3-configobj

# Install here missing dependencies, e.g.
# RUN apt-get install -y python3-skimage

COPY --chown=build . $SRC/$APPNAME
WORKDIR $BUILD/$APPNAME
RUN chown build .
USER build
ARG BUILD_ID
ARG CMAKE_ARGS
RUN cmake $SRC/$APPNAME -DTRIQS_ROOT=${INSTALL} $CMAKE_ARGS && make -j4 || make -j1 VERBOSE=1
USER root
RUN git config --global --add safe.directory $BUILD/$APPNAME/w2dyn_project/src/w2dynamics
RUN make install
