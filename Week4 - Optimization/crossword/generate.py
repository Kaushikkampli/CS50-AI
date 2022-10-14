import queue
import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """

        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """

        for var in self.crossword.variables:
            for word in self.crossword.words:
                if len(word) != var.length:
                    self.domains[var].remove(word)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """

        tampered = False

        intersec = self.crossword.overlaps[x,y]
        ind1 = intersec[0]
        ind2 = intersec[1]
        
        s = self.domains[x].copy()

        for word1 in s:
            
            match = False

            for word2 in self.domains[y]:
                match = match or (word1[ind1] == word2[ind2])
            
            if not match:
                self.domains[x].remove(word1)
                tampered = True

        return tampered

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """

        queue = []

        vars = self.crossword.variables

        for var1 in vars:
            for var2 in vars:
                if var1 != var2:
                    if self.crossword.overlaps[var1,var2] != None:
                        queue.append((var1,var2))

        while len(queue) != 0:
            
            temp = queue.pop()

            x = temp[0]
            y = temp[1]

            if self.revise(x,y):
                
                if len(self.domains[x]) == 0:
                    return False

                for z in vars:
                    if x != z and y != z and self.crossword.overlaps[z,x] != None:
                        queue.append((z,x))

        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        vars = self.crossword.variables

        for var in vars:
            if var not in assignment.keys():
                return False
        
        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """

        s = set()

        for key in assignment.keys():
            if assignment[key] not in s:
                s.add(assignment[key])
            else:
                return False

        for x in assignment.keys():
            for y in assignment.keys():
                val1 = assignment[x]
                val2 = assignment[y]
                if x != y and self.crossword.overlaps[x,y] != None:
                    intersec = self.crossword.overlaps[x,y]

                    ind1 = intersec[0]
                    ind2 = intersec[1]

                    if val1[ind1] != val2[ind2]:
                        return False
        return True

    def order_domain_values(self,v, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """

        vars = self.crossword.variables
        constraint = dict()
        l = list(self.domains[v])

        for word in self.domains[v]:
            constraint[word] = 0
            for var in vars:
                if v != var and word in self.domains[var]:
                    n = constraint[word]
                    constraint.update({word:n + 1})


        l.sort(key = lambda x:constraint[x])
        return l

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """

        vars = self.crossword.variables

        if len(vars) - len(assignment.keys()) == 1:
            return list(vars - assignment.keys())[0]
        
        l = list(vars - assignment.keys())
        l.sort(key = lambda x:len(self.domains[x]))


        if len(self.domains[l[0]]) == len(self.domains[l[1]]):

            deg = dict()
            for x in l:
                deg[x] = 0
                for y in l:
                    if x != y and self.crossword.overlaps[x,y]:
                        n = deg[x]
                        deg.update({x:n+1})

            l.sort(key = lambda v:deg[v])     

            
        return l[0]


    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """

        if self.assignment_complete(assignment):
            return assignment

        v = self.select_unassigned_variable(assignment)
        for word in self.order_domain_values(v, assignment):
            assignment[v] = word
            if self.consistent(assignment):
                if self.backtrack(assignment):
                    return assignment
            del assignment[v]   

        return None      

def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()