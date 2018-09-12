DIRS = ./
ODIR = ./

# compiler
CC_SERIAL     =  g++
CC_MPI        =  mpicxx
OMPI_CC       =  mpicxx
OMPI_CLINKER  =  mpicxx
MPI_LIB       =  ${MPI_HOME}
OPTFLAGS	  =  -Wall -O3
CPPFLAGS	  =  -std=c++11

# define any directories containing header files other than /usr/include
#   define library paths in addition to /usr/lib
#   if I wanted to include libraries not in /usr/lib I'd specify
#   their path using -Lpath, something like:
#   define any libraries to link into executable:
#   if I want to link in libraries (libx.so or libx.a) I use the -llibname
#   option, something like (this will link in libmylib.so and libm.so:

LMP_LIB     =  ${SRC}/lammps/src/
CINCLUDE 	=  -I../include  -I${LMP_LIB}
LIBS   		=  -L${LMP_LIB} # -L${MPI_HOME}/lib
CDLINK  	=  ${LIBS} -lm -lnlopt -lmpi -lpthread  -larmadillo -framework Accelerate -lboost_mpi -lboost_serialization
                 
# parallel
PARALLEL = MPI
CC = ${CC_MPI}

MAKETARGET 	=  scan.exe 

# The source files
POTFITSRC 	=  ./prefMinMpi.cpp

			   # use lammps lib
			   #./pfLmpDrv.cpp \
			   ./pfLmpBCC.cpp \
			   ./pfLmpFCC.cpp \
			   ./pfLmpHCP.cpp \
			   ./pfLmpEla.cpp \
			   ./pfLmpSuf.cpp \
			   ./pfLmpSufUrlx.cpp \
			   ./pfLmpGSF.cpp \
			   ./pfLmpGSFUrlx.cpp \
			   ./pfLmpPV.cpp \
			   ./pfLmpVac.cpp \
			   ./pfLmpItenOpt.cpp \
			   ./pfLmpItenRun.cpp \

			   #./pfForceADP.cpp   \

# OPT_FLAGS   += ${${PARALLEL}_FLAGS} ${OPT_${PARALLEL}_FLAGS} -DNDEBUG

###########################################################################
# Rules
###########################################################################

# all objects depend on headers
OBJECTS := $(subst .cpp,.o,${POTFITSRC})

%.o: %.cpp
	${CC} -c  $<  ${CINCLUDE}  ${OPTFLAGS}  ${CPPFLAGS} # ${LIBS}

${MAKETARGET}:$(OBJECTS)

all: ${MAKETARGET}
	${CC}  -o  ${MAKETARGET}  ${OBJECTS}  ${CDLINK}  ${OPTFLAGS}  ${CPPFLAGS}

clean:
	rm -f  *.o  *.u  *~ # \#* *.V *.T *.O *.il
