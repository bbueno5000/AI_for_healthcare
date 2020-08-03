"""
DOCSTRING
"""
import gzip
import math
import editdistance
import numpy
import os
import pandas
import pickle
import rdkit
import string
import sys

class MolDistance:
    """
    DOCSTRING
    """
    def __call__(self):
        strings = list()
        for line in sys.stdin:
            strings.append(line)
        strings = [s.rstrip() for s in strings if not s.isspace()]
        levenshtein_distances = [levenshtein(x, y) for x in strings for y in strings]
        print('Average length', numpy.mean([len(s) for s in strings]))
        print('Tanimoto diversity', mol_diversity(strings))
        max_len = max([len(s) for s in strings])
        strings = [pad(s, max_len) for s in strings]
        hamming_distances = [hamming(x, y) for x in strings for y in strings]
        entropies = [H(s, string.printable) for s in strings]
        print('Mean entropy', numpy.mean(entropies))
        print('Mean pairwise Hamming distance', numpy.mean(hamming_distances))
        print('Mean pairwise Levenshtein distance', numpy.mean(levenshtein_distances))

    def H(self, data, characters):
        """
        DOCSTRING
        """
        if not data:
            return 0
        entropy = 0
        for x in characters:
            p_x = float(data.count(x))/len(data)
            if p_x > 0:
                entropy += - p_x*math.log(p_x, 2)
        return entropy

    def hamming(self, s1, s2):
        """
        DOCSTRING
        """
        assert(len(s1) == len(s2))
        return float(sum(c1 != c2 for c1, c2 in zip(s1, s2))) / len(s1)

    def levenshtein(self, s1, s2):
        """
        DOCSTRING
        """
        return float(editdistance.eval(s1, s2)) / max(len(s1), len(s2))

    def mol_diversity(self, smiles):
        """
        DOCSTRING
        """
        df = pandas.DataFrame({'smiles': smiles})
        PandasTools.AddMoleculeColumnToFrame(df, 'smiles', 'mol')
        fps = [rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(m, 4, nBits=2048) for m in df['mol']]
        dist_1d = tanimoto_1d(fps)
        mean_dist = numpy.mean(dist_1d)
        return mean_dist
        mean_rand = 0.91549 # mean random distance
        mean_diverse = 0.94170 # mean diverse distance
        norm_dist = (mean_dist - mean_rand) / (mean_diverse - mean_rand)
        return norm_dist

    def mol_grid_image(self, smiles, file_name, labels=None, molPerRow=5):
        """
        DOCSTRING
        """
        df = pandas.DataFrame({'smiles': smiles})
        PandasTools.AddMoleculeColumnToFrame(df, 'smiles', 'mol')
        if labels is None:
            labels = ['{:d}'.format(i) for i in df.index]
        svg = rdkit.Chem.Draw.MolsToGridImage(df['mol'], molsPerRow=5, legends=labels, useSVG=True)
        save_svg(svg, file_name + '.svg', dpi=150)
        return   

    def np_score(self, mol, fscore=None):
        """
        DOCSTRING
        """
        if fscore is None:
            fscore =readNPModel()
        if mol is None:
            raise ValueError('invalid molecule')
        fp = rdkit.Chem.rdMolDescriptors.GetMorganFingerprint(mol, 2)
        bits = fp.GetNonzeroElements()
        # calculating the score
        score = 0.
        for bit in bits:
            score += fscore.get(bit, 0)
        score /= float(mol.GetNumAtoms())
        # preventing score explosion for exotic molecules
        if score > 4:
            score = 4.0 + math.log10(score - 4.0 + 1.0)
        if score < -4:
            score = -4.0 - math.log10(-4.0 - score + 1.0)
        return score

    def np_scores(self, smiles):
        """
        DOCSTRING
        """
        fscore = readNPModel()
        df = pandas.DataFrame({'smiles': smiles})
        PandasTools.AddMoleculeColumnToFrame(df, 'smiles', 'mol')
        scores = [ NP_score(m,fscore) for m in df['mol']]
        return scores

    def num_bridgeheads_and_spiro(self, mol, ri=None):
        """
        DOCSTRING
        """
        nSpiro = rdkit.Chem.rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nBridgehead = rdkit.Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        return nBridgehead, nSpiro

    def pad(self, s, max_len, pad_char = '_'):
        """
        DOCSTRING
        """
        if len(s) >= max_len:
            return s
        return s + pad_char * (max_len - len(s))

    def read_fragment_scores(self, name='fpscores'):
        """
        DOCSTRING
        """
        import gzip
        global _fscores
        # generate the full path filename:
        if name == "fpscores":
            name = os.path.join(os.path.dirname(__file__), name)
        _fscores = pickle.load(gzip.open('%s.pkl.gz' % name))
        outDict = {}
        for i in _fscores:
            for j in range(1, len(i)):
                outDict[i[j]] = float(i[0])
        _fscores = outDict    
    
    def read_np_model(self, filename='publicnp.model.gz'):
        """
        DOCSTRING
        """
        sys.stderr.write("reading NP model ...\n")
        fscore = pickle.load(gzip.open(filename))
        sys.stderr.write("model in\n")
        return fscore

    def sa_score(self, m,fscores=None):
        """
        DOCSTRING
        """
        if fscores is None:
            readFragmentScores()
        # fragment score
        fp = rdkit.Chem.rdMolDescriptors.GetMorganFingerprint(
            m, 2) # 2 is the radius of the circular fingerprint
        fps = fp.GetNonzeroElements()
        score1 = 0.0
        nf = 0
        #for bitId, v in fps.items():
        for bitId, v in rdkit.six.iteritems(fps):
            nf += v
            sfp = bitId
            score1 += _fscores.get(sfp, -4) * v
        score1 /= nf
        # features score
        nAtoms = m.GetNumAtoms()
        nChiralCenters = len(rdkit.Chem.AllChem.FindMolChiralCenters(m, includeUnassigned=True))
        ri = m.GetRingInfo()
        nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
        nMacrocycles = 0
        for x in ri.AtomRings():
            if len(x) > 8:
                nMacrocycles += 1
        sizePenalty = nAtoms**1.005 - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = 0.0
        # NOTE: the following differs from the paper, which defines:
        # macrocyclePenalty = math.log10(nMacrocycles+1)
        # This form generates better results when 2 or more macrocycles are present
        if nMacrocycles > 0:
            macrocyclePenalty = math.log10(2)
        score2 = (
            0.0 - sizePenalty - stereoPenalty -
            spiroPenalty - bridgePenalty - macrocyclePenalty)
        # correction for the fingerprint density
        # not in the original publication, added in version 1.1
        # to make highly symmetrical molecules easier to synthetise
        score3 = 0.0
        if nAtoms > len(fps):
            score3 = math.log(float(nAtoms) / len(fps)) * 0.5
        sascore = score1 + score2 + score3
        # need to transform "raw" value into scale between 1 and 10
        min = -4.0
        max = 2.5
        sascore = 110 - (sascore - min + 1) / (max - min) * 9.0
        # smooth the 10-end
        if sascore > 8.0:
            sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
        if sascore > 10.0:
            sascore = 10.0
        elif sascore < 1.0:
            sascore = 1.0
        return sascore

    def sa_scores(self, smiles):
        """
        DOCSTRING
        """
        fscores = readFragmentScores(name='fpscores')
        df = pandas.DataFrame({'smiles': smiles})
        PandasTools.AddMoleculeColumnToFrame(df, 'smiles', 'mol')
        scores = [ SA_score(m,fscores) for m in df['mol']]
        return scores

    def save_svg(self, svg, svg_file, dpi=300):
        """
        DOCSTRING
        """
        png_file = svg_file.replace('.svg', '.png')
        with open(svg_file, 'w') as afile:
            afile.write(svg)
        a_str = svg.encode('utf-8')
        #cairosvg.svg2png(bytestring=a_str, write_to=png_file, dpi=dpi)
        return    
    
    def tanimoto_1d(self, fps):
        """
        DOCSTRING
        """
        ds = list()
        for i in range(1, len(fps)):
            ds.extend(rdkit.DataStructs.BulkTanimotoSimilarity(
                fps[i], fps[:i], returnDistance=True))
        return ds

class MusicDistance:
    """
    DOCSTRING
    """
    def __init__(self):
        files = glob('data/*/epoch_data/*199.abc')

    def __call__(self):
        for path in files:
            with open(path, 'r') as fp:
                def clean(seq): return [c for c in seq if c != '']
                seqs = [clean(seq.strip().split(' ')) for seq in fp.readlines()]
                seqs = seqs[:2000]
                print(len(seqs))
                levenshtein_distances = [levenshtein(x, y) for x in seqs for y in seqs]
                print(path)
                print('Levenshtein distance: {}'.format(np.mean(levenshtein_distances)))

    def levenshtein(self, s1, s2):
        """
        DOCSTRING
        """
        return float(editdistance.eval(s1, s2)) / 80
