#!/usr/bin/env python

from distutils.core import setup

setup(name='Tracer',
      version='0.2',
      description='General Ray-tracing library in Python',
      author='The Tracer developers',
      packages=['tracer', 'tracer.models', 'tracer.CoIn_rendering', 'ray_trace_utils'],
      license="GPL v3.0 or later, see LICENSE file"
     )

