import logging
import os

from llm_sepweight import PartSpec


def test_b():
    assert PartSpec(begin=True, mid=[], end=False, is_full=False) == PartSpec.from_string('b')


def test_e():
    assert PartSpec(begin=False, mid=[], end=True, is_full=False) == PartSpec.from_string('e')


def test_st():
    assert PartSpec(begin=True, mid=[range(0, 5)], end=False, is_full=False) == PartSpec.from_string('b 0-5')


def test_en():
    assert PartSpec(begin=False, mid=[range(55, 60)], end=True, is_full=False) == PartSpec.from_string('55-60 e')


def test_complex():
    assert PartSpec(begin=False, mid=[range(0, 5), range(50, 55)], end=False, is_full=False) == PartSpec.from_string('0-5 50-55')


def test_im():
    assert PartSpec(
        begin=False,
        mid=[range(5, 10), range(15, 20), range(25, 30)],
        end=False,
        is_full=False
    ) == PartSpec.from_string('5-10 15-20 25-30')
