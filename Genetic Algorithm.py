import random
import math
import pickle
import itertools
import sys

class GenePool():

    def __init__(self, sample, population, cross_coefficient=0.1, elitism=0.05, mutation=0.05,
                 max_iterations=-1,save="algebra_2005_2006_train.txt"):
        
        self.sample = sample
        self.population = population
        self.genes = {}
        self.new_gene_pool = []
        self.cross_coefficient = cross_coefficient
        self.elitism = elitism
        self.mutation = mutation
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.total_result = 0
        self.calls = -1
        self.save = save
        self.convergence = False
        self.new_genes()

    # create new genes (using all the data from Algebra I 2005-2006 as the training data)
    def new_genes(self):
        while len(self) != self.population:
            dna = []
            for chromosome in self.sample:
                if chromosome == bool:
                    dna.append(bool(random.getrandbits(1)))
                elif chromosome == int:
                    dna.append(random.randint(0, 2 ** 10))
                elif chromosome == float:
                    dna.append(random.random())
            gene = Genes(self, '%i' % (len(self) + 1), dna)
            self[gene.name] = gene

    def next_gene(self):
        """
        :returns: A gene to be used.
        :rtype: Genes

        """
        self.calls += 1
        if self.calls >= self.population:
            self.calls = 0
            self.start_genetic_algorithm()

        return self[self.calls]

    def replace_gene(self, old_gene, new_dna, result, rank):
         new_gene = Genes(self, old_gene.name, new_dna, result=result, rank=rank)
         self[old_gene.name] = new_gene
         self.new_gene_pool.append(new_gene)

    def rated_list(self, cut):
        rated_list = sorted([gene for gene in self.genes.items()])[::-1]

        selected = int(math.ceil(len(self) * self.elitism))
        top_genes = rated_list[0:selected]
        mid_genes = rated_list[selected:len(self) - selected]

        genes = None
        if cut == 'top':
            genes = top_genes
        elif cut == 'mid':
            genes = mid_genes

        return genes

    def duplicate_top(self):
        top_genes = self.rated_list('top')
        for gene in top_genes:
            gene = gene[1]
            gene.rank /= 2
            gene.result /= 2
            new_gene = Genes(self, "%i" % (len(self) + 1), gene.dna, gene.result, gene.rank)
            self[new_gene.name] = new_gene
            self.new_gene_pool.append(new_gene)
            gene.lock = True
            self.new_gene_pool.append(gene)

    def mutate_mid_genes(self):
        for gene in self:
            new_dna = []
            for chromosome in gene.dna:
                rand = random.random()
                if rand <= self.mutation:
                    if type(chromosome) == bool:
                        new_dna.append(bool(random.getrandbits(1)))
                    if type(chromosome) == int:
                        new_dna.append(random.randint(0, 2 ** 10))
                    if type(chromosome) == float:
                        new_dna.append(random.random())
                else:
                    new_dna.append(chromosome)
            gene.dna = new_dna

    def cross_genes(self):
        ranks = [gene.rank for gene in self.genes.values()]
        min_rank = min(ranks)
        max_rank = max(ranks)
        max_tries = 0
        while len(self.new_gene_pool) < self.population or max_tries == 100:
            max_tries += 1
            gene1 = None
            gene2 = None
            while gene1 is None or gene2 is None:
                rand = random.uniform(min_rank, max_rank)
                gene1 = [gene for gene in self.genes.values() if gene.rank >= rand and not gene.lock and gene
                    .rank != 1]
                if gene1:
                    gene1 = random.choice(gene1)
                    pass
                else:
                    gene1 = None
                rand = random.uniform(min_rank, max_rank)
                gene2 = [gene for gene in self.genes.values() if gene.rank >= rand and not gene.lock and gene !=
                                                                     gene1 and gene.rank != 1]
                if gene2:
                    gene2 = random.choice(gene2)
                else:
                    gene2 = None
            self.crossover(gene1, gene2)
        for gene in self:
            gene.lock = False

    def crossover(self, gene1, gene2):
        """Two-Point cross-over"""
        cross_point1, cross_point2 = None, None
        while not cross_point2 and not cross_point1:
            cross_point1 = random.randint(0, len(gene1))
            cross_point2 = random.randint(0, len(gene1))
            if cross_point1 == cross_point2 or cross_point1 > cross_point2:
                cross_point1, cross_point2 = None, None
        new_dna1 = gene1[:cross_point1] + gene2[cross_point1:cross_point2] + gene1[cross_point2:]
        new_dna2 = gene2[:cross_point1] + gene1[cross_point1:cross_point2] + gene2[cross_point2:]
        result = (gene1.result + gene2.result) / 2
        rank = (gene1.rank + gene2.rank) / 2
        self.replace_gene(gene1, new_dna1, result, rank)
        self.replace_gene(gene2, new_dna2, result, rank)

    def return_result(self, gene, result):
        """Return here the gene, it's fitness result. Set force to True to run the GA in the gene_pool
        :param gene: The gene which was run.
        :type gene: Genes
        :param result: The fitness result.
        :type result: int
        """
        gene.result += result
        self.total_result += gene.result

    def test_convergence(self, print_result=False):
        if print_result:
            pro = 0
            con = 0
            for gene1, gene2 in itertools.combinations(self.genes.values(), 2):
                if gene1.dna == gene2.dna:
                    pro += 1
                else:
                    con += 1
            cc = (con + 1) / (pro + 1)
            print ("Converged rate: 1:%i" % cc)

        if self.convergence or self.current_iteration == self.max_iterations:
            for gene in self:
                print ("Convergence Found: \nGene: %s\nDNA: %s" % (gene.name, gene.dna),
                sys.exit("End of the line"))

    def reset_genes(self):
        for gene in self:
            gene.rank = 0
            gene.result = 0
            gene.lock = False
        self.total_result = 0

    def save_results(self):
        f = open(self.save, 'w')
        pickle.dump(self, f)
        f.close()

    def update_result(self):
        self.total_result = 0
        self.genes = {}
        for gene in self.new_gene_pool:
            new_gene = Genes(self, '%i' % (len(self.genes) + 1), gene.dna)
            self.genes[new_gene.name] = new_gene
        self.new_gene_pool = []

    def start_genetic_algorithm(self):
        [gene.update_rank() for gene in self.genes.values()]
        self.current_iteration += 1
        self.duplicate_top()
        self.cross_genes()
        self.update_result()
        self.test_convergence(print_result=True)
        self.mutate_mid_genes()
        # self.saveResults()
        return self.convergence

    def __getitem__(self, item):
        if type(item) == str:
            return self.genes[item]
        if type(item) == int:
            return self[sorted(self.genes)[item]]

    def __setitem__(self, key, values):
        self.genes[key] = values

    def __len__(self):
        return len(self.genes)


class Genes():
    """
    A gene which mutates based on it's gene_pool settings

    :param gene_pool: Where all genes are stored and the GA will run
    :type gene_pool: GenePool
    :param dna: The core of the gene, it's based on the sample set on the gene_pool.
    :type dna: list
    :param result: The current fitness result of the gene.
    :type result: int
    :param rank: The result from 0.0 to 1.0 of the gene in the gene_pool.
    :type rank: float
    """

    def __init__(self, gene_pool, name, dna, result=0, rank=0):
        """should be instantiated by genePool.newGenes()"""
        self.name = name
        self.gene_pool = gene_pool
        self.dna = dna
        self.result = result
        self.rank = rank
        self.lock = False

    def update_rank(self):
        self.rank = self.result / float(self.gene_pool.total_result)
        return self.rank

    def __len__(self):
        return len(self.dna)

    def __getitem__(self, item):
        return self.dna[item]

gene_pool = GenePool([int] * 3, 100)

while True:
    gene = gene_pool.next_gene()
    dna = gene.dna
    result = 1
    if dna[0] <= 100:
        result += 10
        if dna[0] == 100:
            result += 100
    if dna[1] >= 200:
        result += 10
        if dna[1] == 200:
            result += 100
    if 100 <= dna[2] <= 200:
        result += 10
        if dna[2] == 150:
            result += 100

    if dna[0] == 100 and dna[1] == 200 and dna[2] == 150:
        gene_pool.convergence = True

    gene_pool.return_result(gene, result)


