# -*- coding: utf-8 -*-

from constraint import *

#nomi laboratori di analisi
A, B, C, D = "Laboratorio Analisi istologiche A", "Laboratorio Analisi istologiche B", "Laboratorio Analisi istologiche C", "Laboratorio Analisi istologiche D"

#funzione da avviare per la prenotazione
def lab_booking():
    labs = create_problems()
    print("Vuoi prenotare una visita istologica presso un laboratorio di analisi? [si/no]")
    response = str(input())
    
    if response == "si":
        print("[0] %s\n[1] %s\n[2] %s\n[3] %s\n" %(A,B,C,D))
        print("Seleziona il laboratorio: [0/1/2/3]")
        choice = int(input())
        while choice < 0 or choice > 3:
            print("Scelta non valida. Seleziona il laboratorio: [0/1/2/3]")
            choice = int(input())
            
        lab_selected = labs[choice]
        
        first, last = lab_selected.get_availability()
        
        print("Seleziona un turno inserendo il numero del turno associato:")
        turn = int(input())
        while turn < first or turn > last:
            print("Scelta non valida. Seleziona un turno inserendo il numero del turno associato:")
            turn = int(input())
            
        lab_selected.print_single_availability(turn)
        
        
        

#crea i problemi csp con i vari vincoli
def create_problems():
    lab_a = lab_csp(A)
    lab_a.addConstraint(lambda day,hours: hours >= 8 and hours <= 14 if day == "lunedi" else hours >= 15 and hours <= 20 if day == "giovedi" else None ,["day","hours"])

    lab_b = lab_csp(B)
    lab_b.addConstraint(lambda day,hours: hours >= 9 and hours <= 13 if day == "martedi" else hours >= 18 and hours <= 21 if day == "venerdi" else hours >= 10 and hours <= 11 if day == "sabato" else None ,["day","hours"])

    lab_c = lab_csp(C)
    lab_c.addConstraint(lambda day,hours: hours >= 10 and hours <= 14 if day == "mercoledi" else hours >= 8 and hours <= 11 if day == "venerdi" else hours >= 15 and hours <= 17 if day == "sabato" else None ,["day","hours"])

    lab_d = lab_csp(D)
    lab_d.addConstraint(lambda day,hours: hours >= 7 and hours <= 12 if day == "giovedi" else hours >= 12 and hours <= 14 if day == "sabato" else None ,["day","hours"])
    
    return [lab_a, lab_b, lab_c, lab_d]




#classe per csp
class lab_csp(Problem):

    def __init__(self,lab_name: str, solver=None):
        super().__init__(solver=solver)
        self.lab_name = lab_name
        self.days = self.addVariable("day",["lunedi","martedi","mercoledi","giovedi","venerdi","sabato"])
        self.hours = self.addVariable("hours",[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
        self.availability = None





    def get_availability(self):
        self.availability = sorted(self.getSolutions(), key=lambda h: h['hours'])
        first_turn = None
        last_turn = None

        if len(self.availability) > 0:

            print("Disponibilita' laboratorio confermata.\n")
            i = 0
            first_turn = i

            while i < len(self.availability):
                
                print("Turno [%d], Giorno: %s, Orario: %d"%(i,self.availability[i]['day'],self.availability[i]['hours']))
                i = i + 1
            
            last_turn = i-1
            print("\n")
               
        else:
            print("Non c'è disponibilita' per il laboratorio")

        return first_turn, last_turn
    
    
    
    
    
    def print_single_availability(self, index):
        if index >= 0 and index < len(self.availability):
            print("Turno selezionato: [%d], Giorno: %s, Orario: %d\n\n"%(index,self.availability[index]['day'],self.availability[index]['hours']))

