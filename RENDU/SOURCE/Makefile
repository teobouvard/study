#Outils
CPP=g++
LD=g++
ECHO=echo

#Options
CPPFLAGS=-ansi -pedantic -Wall -std=c++11 -g #-DMAP2
RMFLAGS=-rf
RM=rm

#Fichiers
HEAD=$(wildcard *.h)
SRC=$(wildcard *.cpp)
OBJ=$(SRC:.cpp=.o)

#Cibles spéciales
EXE=out
RUN=run
CLEAN=clean

.PHONY: $(CLEAN) $(RUN)

#Cibles
$(EXE): $(OBJ)
	@$(ECHO) "Edition des liens"
	@$(LD) $^ -o $@

$(RUN): $(EXE)
	@$(ECHO) "Execution de $^"
	@$../$(EXE)

$(CLEAN):
	@$(RM) $(RMFLAGS) $(EXE) $(OBJ)

#Regles d'inférences
%.o: %.cpp %.h
	@$(ECHO) "Compilation de $<"
	@$(CPP) $(CPPFLAGS) -c $< -o $@
