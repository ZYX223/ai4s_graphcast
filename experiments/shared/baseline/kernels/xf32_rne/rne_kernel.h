#pragma once
#include <stddef.h>
extern "C" int graphcast_round_into(const float*,float*,int,int,int,void*);
extern "C" const float* graphcast_round_scratch(const float*,int,int,int,void*,int);
