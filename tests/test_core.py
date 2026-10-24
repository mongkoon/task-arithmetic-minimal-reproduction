"""Deterministic offline unit tests; no pretrained model downloads required."""
import math
import pytest
import torch
from task_arithmetic.core import ArithmeticModel, extract_vector, vector_cosine, vector_norm, save_vector, load_vector


def linear(values):
    m = torch.nn.Linear(2, 1, bias=True)
    with torch.no_grad():
        m.weight.copy_(torch.tensor([values[:2]], dtype=torch.float32))
        m.bias.fill_(values[2])
    return m


def test_alpha_one_reconstructs_fine_tuned_and_restores():
    base = linear([1,2,3])
    fine = linear([2,4,6])
    delta = extract_vector(base, fine)
    arithmetic = ArithmeticModel(base)
    arithmetic.register("task", delta)
    arithmetic.apply({"task":1.0})
    assert arithmetic.max_abs_error_vs(fine) < 1e-6
    arithmetic.restore()
    assert torch.allclose(base.weight, torch.tensor([[1.,2.]]))


def test_multiple_applications_do_not_accumulate():
    base=linear([1,2,3]);fine=linear([2,4,6]);v=extract_vector(base,fine)
    editor=ArithmeticModel(base);editor.register("a",v)
    editor.apply({"a":0.5});first=base.weight.detach().clone()
    editor.apply({"a":0.5})
    assert torch.equal(base.weight,first)
    editor.apply({"a":-1})
    assert torch.allclose(base.weight,torch.tensor([[0.,0.]]))


def test_two_vector_addition_and_analogy():
    base=linear([0,0,0]);fine_a=linear([1,2,3]);fine_b=linear([4,5,6]);fine_c=linear([7,8,9])
    editor=ArithmeticModel(base)
    for name,other in [("a",fine_a),("b",fine_b),("c",fine_c)]:
        editor.register(name,extract_vector(base,other))
    editor.apply({"a":1,"b":1})
    assert torch.allclose(base.weight,torch.tensor([[5.,7.]]))
    editor.apply({"a":-1,"b":1,"c":1})
    assert torch.allclose(base.weight,torch.tensor([[10.,11.]]))


def test_incompatible_vector_rejected():
    base=linear([0,0,0]);editor=ArithmeticModel(base)
    with pytest.raises(ValueError,match="parameter mismatch"):
        editor.register("bad",{"wrong":torch.ones(2)})
    with pytest.raises(KeyError):
        editor.apply({"not_found":1})


def test_vector_file_metadata(tmp_path):
    base=linear([0,0,0]);fine=linear([1,2,3])
    delta=extract_vector(base,fine)
    path=tmp_path/"a.pt"
    save_vector(delta,path,{"base_model":"demo"})
    reloaded,meta=load_vector(path,expected_base="demo")
    assert meta["base_model"]=="demo"
    assert math.isclose(vector_cosine(delta,reloaded),1.0)
    assert vector_norm(reloaded)>0
    with pytest.raises(ValueError,match="expects base"):
        load_vector(path,expected_base="different")


def test_tied_parameters_not_counted_twice():
    class Shared(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb=torch.nn.Embedding(2,2)
            self.decoder=torch.nn.Linear(2,2,bias=False)
            self.decoder.weight=self.emb.weight
    model=Shared()
    editor=ArithmeticModel(model)
    assert len(editor.base)==1
