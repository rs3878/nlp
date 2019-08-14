"""
COMS W4705 - Natural Language Processing - Summer 19 
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""

import sys
from collections import defaultdict
from math import fsum
import numpy as np

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";", 1)
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        for k,v in self.lhs_to_rules.items():
            for j in range(len(v)):

                if k != v[j][0]:
                    print("Error in key of lhs")
                    return False

                if len(v[j][1]) == 2:
                    if v[j][0].isupper() is False:
                        return False
                    for i in range(2):
                        if v[j][1][i].upper() != v[j][1][i]:
                            print("error in non-terminal", v[j][1])
                            return False

                if len(v[j][1]) == 1:
                    if v[j][1][0].lower() != v[j][1][0]:
                        print("error in terminal", v[j][1])
                        return False

        for k in self.lhs_to_rules.keys():
            prob = fsum(np.float128(p[-1]) for p in self.lhs_to_rules[k])
            if np.abs(prob - 1.0) > 1e-11:
                print("prob not sum to 1: ", prob)
                return False

        for k,v in self.rhs_to_rules.items():
            for j in range(len(v)):

                if k != v[j][1]:
                    print("Error in key of rhs")
                    return False

                if len(v[j][1]) == 2:
                    if v[j][0].isupper() is False:
                        return False
                    for i in range(2):
                        if v[j][1][i].upper() != v[j][1][i]:
                            print("error in non-terminal", v[j][1])
                            return False

                if len(v[j][1]) == 1:
                    if v[j][1][0].lower() != v[j][1][0]:
                        print("error in terminal", v[j][1])
                        return False

        return True


if __name__ == "__main__":
    with open(sys.argv[1],'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        print(grammar.verify_grammar())
