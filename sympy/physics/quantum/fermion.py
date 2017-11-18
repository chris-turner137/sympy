"""Fermionic quantum operators."""

from sympy import Mul, Add, Integer
from sympy.core.function import expand
from sympy.physics.quantum import Operator
from sympy.physics.quantum import HilbertSpace, Ket, Bra
from sympy.functions.special.tensor_functions import KroneckerDelta
import bisect

__all__ = [
    'FermionOp',
    'FermionFockKet',
    'FermionFockBra',
    'qsimplify_fermion'
]


class FermionOp(Operator):
    """A fermionic operator that satisfies {c, Dagger(c)} == 1.

    Parameters
    ==========

    name : int
        An optional integer that labels the fermionic mode. Fermionic operators
        with different names anti-commute.

    annihilation : bool
        A bool that indicates if the fermionic operator is an annihilation
        (True, default value) or creation operator (False)

    Examples
    ========

    >>> from sympy.physics.quantum import Dagger, AntiCommutator
    >>> from sympy.physics.quantum.fermion import FermionOp
    >>> c = FermionOp()
    >>> AntiCommutator(c, Dagger(c)).doit()
    1
    """
    @property
    def name(self):
        return self.args[0]

    @property
    def is_annihilation(self):
        return bool(self.args[1])

    @classmethod
    def default_args(self):
        return (0, True)

    def __new__(cls, *args, **hints):
        if not len(args) in [1, 2]:
            raise ValueError('1 or 2 parameters expected, got %s' % args)

        if len(args) == 1:
            args = (args[0], Integer(1))

        if len(args) == 2:
            args = (args[0], Integer(args[1]))

        return Operator.__new__(cls, *args)

    def _eval_commutator_FermionOp(self, other, **hints):
        if 'independent' in hints and hints['independent']:
            # [c, d] = 0
            return Integer(0)

        return None

    def _eval_anticommutator_FermionOp(self, other, **hints):
        if self.name == other.name:
            # {a^\dagger, a} = 1
            if not self.is_annihilation and other.is_annihilation:
                return Integer(1)

        elif 'independent' in hints and hints['independent']:
            # {c, d} = 2 * c * d, because [c, d] = 0 for independent operators
            return 2 * self * other

        return None

    def _eval_anticommutator_BosonOp(self, other, **hints):
        # because fermions and bosons commute
        return 2 * self * other

    def _eval_commutator_BosonOp(self, other, **hints):
        return Integer(0)

    def _eval_adjoint(self):
        return FermionOp(self.name, not self.is_annihilation)

    def _eval_power(self, e):
        if e.is_Integer and e.is_positive:
            return Integer(0)

    def _print_contents_latex(self, printer, *args):
        if self.is_annihilation:
            return r'{c_{%s}}' % str(self.name)
        else:
            return r'{c_{%s}^\dagger}' % str(self.name)

    def _print_contents(self, printer, *args):
        if self.is_annihilation:
            return r'FermionOp(%s)' % str(self.name)
        else:
            return r'Dagger(FermionOp(%s))' % str(self.name)

    def _print_contents_pretty(self, printer, *args):
        from sympy.printing.pretty.stringpict import prettyForm
        pform = printer._print(self.args[0], *args)
        if self.is_annihilation:
            return pform
        else:
            return pform**prettyForm(u'\N{DAGGER}')


class FermionFockKet(Ket):
    """Fock state ket for a fermionic mode.

    Parameters
    ==========

    n : Number
        The Fock state number.

    """

    def __new__(cls, n):
        if n not in [0, 1]:
            raise ValueError("n must be 0 or 1")
        return Ket.__new__(cls, n)

    @property
    def n(self):
        return self.label[0]

    @classmethod
    def dual_class(self):
        return FermionFockBra

    @classmethod
    def _eval_hilbert_space(cls, label):
        return HilbertSpace()

    def _eval_innerproduct_FermionFockBra(self, bra, **hints):
        return KroneckerDelta(self.n, bra.n)

    def _apply_operator_FermionOp(self, op, **options):
        if op.is_annihilation:
            if self.n == 1:
                return FermionFockKet(0)
            else:
                return Integer(0)
        else:
            if self.n == 0:
                return FermionFockKet(1)
            else:
                return Integer(0)


class FermionFockBra(Bra):
    """Fock state bra for a fermionic mode.

    Parameters
    ==========

    n : Number
        The Fock state number.

    """

    def __new__(cls, n):
        if n not in [0, 1]:
            raise ValueError("n must be 0 or 1")
        return Bra.__new__(cls, n)

    @property
    def n(self):
        return self.label[0]

    @classmethod
    def dual_class(self):
        return FermionFockKet

def _qsimplify_fermion_product_site(e):
    """
    Assumes that all fermion operators have the same name.
    """
    work = [e]
    out = []
    while len(work):
        e = work.pop()
        if not isinstance(e, Mul):
            out.append(e)
            continue

        c, nc = e.args_cnc()
        ann = [f.is_annihilation for f in nc]

        # Find the first inverted pair of operators
        idx = next((i for i in range(len(ann)) if ann[i:i+2] == [True,False]), None)
        if idx is None:
            out.append(e)
            continue

        # Substitute the inverted pair for a cannonical subexpression
        e = Mul(*c) * (Mul(*nc[:idx])
                       * (Integer(1) - Mul(nc[idx+1],nc[idx]))
                       * Mul(*nc[idx+2:]))

        # Expand the new as a sum of fermion operator strings
        e = expand(e)

        # Recursive simplify
        if isinstance(e, Add):
            work.extend(e.args)
        else:
            work.append(e)
    return Add(*out)

def qsimplify_fermion(e, full=False):
    if isinstance(e, Operator):
        return e

    if isinstance(e, (Add)):
        t = type(e)
        return t(*[qsimplify_fermion(arg, full=full) for arg in e.args])

    if not isinstance(e, Mul):
        return e

    c, nc = e.args_cnc()

    nc_s = []
    while nc:
        if not isinstance(nc[0], FermionOp):
            nc_s.append(nc.pop(0))
            continue

        # Permute pauli strings into a name sorted order
        ops = [nc.pop(0)]
        names = [ops[0].name]
        while (len(nc) and isinstance(nc[0], FermionOp)):
            x = nc.pop(0)
            idx = bisect.bisect_right(names, x.name)
            sign = (-1)**(len(ops) - idx)
            ops.insert(idx, x)
            names.insert(idx, x.name)
            c = c + [sign]

        if full:
            # Permute into normal order form
            lo = 0
            while len(names)-lo:
                hi = bisect.bisect_right(names, names[lo], lo=lo)
                xs = ops[lo:hi]
                nc_s.append(_qsimplify_fermion_product_site(Mul(*xs)))
                lo = hi

        else:
            nc_s.extend(ops)

    return Mul(*c) * Mul(*nc_s)
