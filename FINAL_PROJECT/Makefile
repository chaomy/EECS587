
#PFTOOLS   = 
#CINCLUDE  =   

CXX       =  nvcc 
CPPFLAGS +=  -Wall -g -O3 -std=c++11
CDLINK    =  -lm 

SRCS = qmInput.cu \
	   qmUpdate.cu \
	   qmMain.cu  

MAKETARGET 	= exc_qm 

OBJECTS := $(subst .cu,.o,${SRCS})

CC = ${CXX}

%.o: %.cpp
	${CC} -c  $<  ${CINCLUDE} ${CPPFLAGS}  #

${MAKETARGET}:$(OBJECTS)


all: ${MAKETARGET}
	${CC}  -o  ${MAKETARGET}  ${OBJECTS}  ${CDLINK}  ${CPPFLAGS}

clean:
	rm -f *.o