"""Tests for the wip package."""

import pytest

from wip import hello


def test_hello_default():
    """Test hello function with default parameter."""
    result = hello()
    assert result == "Hello, World!"


def test_hello_with_name():
    """Test hello function with custom name."""
    result = hello("Alice")
    assert result == "Hello, Alice!"


def test_hello_with_empty_string():
    """Test hello function with empty string."""
    result = hello("")
    assert result == "Hello, !"
