#Outils
CPP=g++
LD=g++
ECHO=echo

#Options
CPPFLAGS=-ansi -pedantic -Wall -std=c++11 -DMAP
RMFLAGS=-rf
LDFLAGS=


#Fichiers
HEAD=$(wildcard *.h)
SRC=$(wildcard *.cpp)
OBJ=$(SRC:.cpp=.o)
LIBS=

#Cibles spéciales
EXE=out
RUN=run
CLEAN=clean

.PHONY: $(CLEAN) $(RUN)

#Cibles
$(EXE): $(OBJ)
	@$(ECHO) "Edition des liens"
	@$(LD) $(LDFLAGS) $^ -o $@ $(LIBS)

$(RUN): $(EXE)
	@$(ECHO) "Execution de $^"
	@$../$(EXE)

$(CLEAN):
	@$(RM) $(RMFLAGS) $(EXE) $(OBJ)

#Regles d'inférences
%.o: %.cpp
	@$(ECHO) "Compilation de $<"
	@$(CPP) $(CPPFLAGS) -c $< -o $@
