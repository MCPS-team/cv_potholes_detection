FROM python:3.7

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install numpy

WORKDIR /
ENV OPENCV_VERSION="4.1.1"
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
&& unzip ${OPENCV_VERSION}.zip \
&& mkdir /opencv-${OPENCV_VERSION}/cmake_binary \
&& cd /opencv-${OPENCV_VERSION}/cmake_binary \
&& cmake -DBUILD_TIFF=ON \
  -DBUILD_opencv_java=OFF \
  -DWITH_CUDA=OFF \
  -DOPENCV_GENERATE_PKGCONFIG=YES \
  -DWITH_OPENGL=ON \
  -DWITH_OPENCL=ON \
  -DWITH_IPP=ON \
  -DWITH_TBB=ON \
  -DWITH_EIGEN=ON \
  -DWITH_V4L=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DCMAKE_INSTALL_PREFIX=$(python3.7 -c "import sys; print(sys.prefix)") \
  -DPYTHON_EXECUTABLE=$(which python3.7) \
  -DPYTHON_INCLUDE_DIR=$(python3.7 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
  -DPYTHON_PACKAGES_PATH=$(python3.7 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
  .. \
&& make -j8 install \
&& rm /${OPENCV_VERSION}.zip \
&& rm -r /opencv-${OPENCV_VERSION}
RUN ln -s \
  /usr/local/python/cv2/python-3.7/cv2.cpython-37m-x86_64-linux-gnu.so \
  /usr/local/lib/python3.7/site-packages/cv2.so
RUN pip install scikit-image

RUN apt-get update
RUN mkdir cv_potholes_detection

COPY ./ cv_potholes_detection
RUN cd 'cv_potholes_detection/Pothole_Detection/Pothole_Detection_YOLO/darknet' && \
    sh ./Makefile_to_CPU.sh && \
    cat Makefile && \
    make -j8 && \
    bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf' && \
    ldconfig && \
    ln -s '/cv_potholes_detection/Pothole_Detection/Pothole_Detection_YOLO/darknet/libdarknet.so' '/cv_potholes_detection/'
RUN pip install -r /cv_potholes_detection/requirements.txt

CMD cd cv_potholes_detection &&  \
    python main.py





   
