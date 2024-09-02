#!/bin/sh
fmt = "Key,Value\nElapsed Time,%E\nAverage Mem Used(in KB),%K\nPercentage of CPU,%P\nCPU used by Kernel(s),%S\nCPU used in User mode(s),%U\nElapsed Wall Clock Time(s),%e"
cmd = ./scripts/inference/infer_text2natsql.sh base spider
/bin/time -f $fmt $cmd
