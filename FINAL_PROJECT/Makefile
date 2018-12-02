
#PFTOOLS   = 
#CINCLUDE  =   

CXX       =  nvcc 
CPPFLAGS +=  -g -O3 -std=c++11 -arch=sm_35 -rdc=true -lcudadevrt
CDLINK    =  -lm 

SRCS = qmInput.cu \
	   qmUpdate.cu \
	   qmMain.cu  

MAKETARGET 	= exc_qm 

OBJECTS := $(subst .cu,.o,${SRCS})

%.o: %.cu
	${CXX} -c  $<  ${CINCLUDE} ${CPPFLAGS}  #

${MAKETARGET}:$(OBJECTS)

all: ${MAKETARGET}
	${CXX}  -o  ${MAKETARGET}  ${OBJECTS}  ${CDLINK}  ${CPPFLAGS}

clean:
	rm -f *.o