#!/bin/bash

nvcc $1 && nvprof ./a.out
