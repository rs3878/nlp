"""
COMS W4705 - Natural Language Processing - Summer 19 
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""
import math
import sys
from collections import defaultdict
import itertools
import numpy as np
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self, tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        d = {}
        m = len(tokens)

        for i in range(m):
            d[(i,i)] = []
            for item in self.grammar.rhs_to_rules[(tokens[i],)]:
                d[(i,i)].append(item[0])

        for diff in range(1,m):
            for i in range(m):
                if i+diff < m:
                    d[(i,i+diff)] = []

                    j = i
                    while j< i+diff:
                        if d.get((i,j)) != [] and d.get((j+1,i+diff)) != []:
                            for item1 in d[(i,j)]:
                                for item2 in d[(j+1,i+diff)]:
                                    if self.grammar.rhs_to_rules.get((item1, item2)) != None :
                                        for item in self.grammar.rhs_to_rules.get((item1, item2)):
                                            d[(i, i+diff)].append(item[0])
                        j+=1
        if d[0,m-1] != []:
            return True
        else:
            return False
       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        d = {}
        table = {}
        probs = {}
        m = len(tokens)

        for i in range(m):
            d[(i,i+1)] = {}
            probs[(i,i+1)] = {}
            for item in self.grammar.rhs_to_rules[(tokens[i],)]:
                d[(i,i+1)][item[0]]=tokens[i]
                probs[(i,i+1)][item[0]]=np.log(item[-1])

        for diff in range(2,m+1):
            for i in range(m):
                if i+diff <= m:
                    d[(i,i+diff)] = {}
                    probs[(i,i+diff)] = {}
                    j = i+1
                    while j< i+diff:
                        if d.get((i,j)) != {} and d.get((j,i+diff)) != {}:
                            for item1 in d[(i,j)]:
                                for item2 in d[(j,i+diff)]:
                                    if self.grammar.rhs_to_rules.get((item1, item2)) != None :
                                        for item in self.grammar.rhs_to_rules.get((item1, item2)):
                                            try:
                                                score = probs[(i, i + diff)][item[0]]
                                                new_score = probs[(i,j)][item1] + probs[(j,i+diff)][item2] +np.log(item[-1])
                                                if new_score > score:
                                                    probs[(i, i + diff)][item[0]] = new_score
                                                    d[(i,i+diff)][item[0]] = ((item1,i,j),(item2,j,i+diff))
                                            except:
                                                d[(i, i + diff)][item[0]] = ((item1, i, j), (item2, j, i + diff))
                                                probs[(i, i + diff)][item[0]] = probs[(i, j)][item1] + \
                                                                                probs[(j, i + diff)][item2] + \
                                                                                np.log(item[-1])

                        j+=1

        return d, probs


def get_tree(chart, i,j, nt):
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    table = chart
    item = table[(i,j)][nt]
    if type(item) == tuple:
        return (nt,get_tree(table,item[0][1],item[0][2],item[0][0]),get_tree(table,item[1][1],item[1][2],item[1][0]))
    else:
        return (nt, item)
       
if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        print("---------Testing Parser----------")
        toks =['flights', 'from','miami', 'to', 'cleveland','.']
        print("Should be True:", parser.is_in_language(toks))
        toks = ['miami', 'flights', 'cleveland', 'from', 'to', '.']
        print("Should be False:", parser.is_in_language(toks))
        print("\n\n---------Testing Parser with Backpointers---------")
        toks =['flights', 'from','miami', 'to', 'cleveland','.']
        table,probs = parser.parse_with_backpointers(toks)
        assert check_table_format(table)
        assert check_probs_format(probs)
        print("Table is ", table[(0,len(toks))])
        print("Prob is ", probs[(0,len(toks))])
        print("Entire Prob table is ", probs)
        print("\n\n---------Testing Get Tree---------")
        toks = ['flights', 'from', 'miami', 'to', 'cleveland', '.']
        print(table)
        result = get_tree(table, 0, len(toks), grammar.startsymbol)
        correct = ('TOP', ('NP', ('NP', 'flights'), ('NPBAR', ('PP', ('FROM', 'from'), ('NP', 'miami')), ('PP', ('TO', 'to'), ('NP', 'cleveland')))), ('PUN', '.'))
        print(result)
        print(correct)
        print(result == correct)
        print("\n\n---------Testing Get Tree 2---------")
        toks = ['with', 'the', 'least', 'expensive', 'fare', '.']
        table,probs = parser.parse_with_backpointers(toks)
        if probs[(0,len(toks))]:
            result = get_tree(table, 0, len(toks), grammar.startsymbol)
        else:
            print("Not possible")
