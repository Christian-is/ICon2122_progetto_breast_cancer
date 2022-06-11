# -*- coding: utf-8 -*-

import owlready2 as owl
import os

def ontology_analyzer():
    print("ONTOLOGIA\n")
    onto = owl.get_ontology("breast_ontology.owl").load()
    
    #stampo il contenuto principale della ontologia
    print("Lista classi nella ontologia:\n")
    print(list(onto.classes()), "\n")
    
    print("Lista Person nella ontologia:\n")
    persons = onto.search(is_a = onto.Person)
    print(persons, "\n")
    
    print("Lista Cancer nella ontologia:\n")
    cancers = onto.search(is_a = onto.Cancer)
    print(cancers, "\n")
    
    print("Lista Breast_cancer nella ontologia:\n")
    b_cancers = onto.search(is_a = onto.Breast_cancer)
    print(b_cancers, "\n")
    
    print("Lista Analysis nella ontologia:\n")
    analysis = onto.search(is_a = onto.Analysis)
    print(analysis, "\n")
    
    #esempio di query
    print("Lista di persone che hanno un cancro al seno:\n")
    patients = onto.search(is_a = onto.Person, has_breast_cancer = "*")
    print(patients, "\n")
    