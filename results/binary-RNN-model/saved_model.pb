▓К
ы─
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
ђ
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resourceѕ
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
є
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ѕ
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ

NoOp
:
OnesLike
x"T
y"T"
Ttype:
2	

M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
Ц
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	ѕ
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_typeіьout_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕе
@
StaticRegexFullMatch	
input

output
"
patternstring
э
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
░
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleіжУelement_dtype"
element_dtypetype"

shape_typetype:
2	
Ъ
TensorListReserve
element_shape"
shape_type
num_elements(
handleіжУelement_dtype"
element_dtypetype"

shape_typetype:
2	
ѕ
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint         
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
░
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ
ћ
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
ѕ"serve*2.15.12v2.15.0-11-g63f5a65c7cd8йс
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
|
count_1VarHandleOp*
_output_shapes
: *

debug_name
count_1/*
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
|
total_1VarHandleOp*
_output_shapes
: *

debug_name
total_1/*
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
ц
Adam/v/dense_5/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/v/dense_5/bias/*
dtype0*
shape:*$
shared_nameAdam/v/dense_5/bias
w
'Adam/v/dense_5/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_5/bias*
_output_shapes
:*
dtype0
ц
Adam/m/dense_5/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/m/dense_5/bias/*
dtype0*
shape:*$
shared_nameAdam/m/dense_5/bias
w
'Adam/m/dense_5/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_5/bias*
_output_shapes
:*
dtype0
«
Adam/v/dense_5/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/dense_5/kernel/*
dtype0*
shape
:@*&
shared_nameAdam/v/dense_5/kernel

)Adam/v/dense_5/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_5/kernel*
_output_shapes

:@*
dtype0
«
Adam/m/dense_5/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/dense_5/kernel/*
dtype0*
shape
:@*&
shared_nameAdam/m/dense_5/kernel

)Adam/m/dense_5/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_5/kernel*
_output_shapes

:@*
dtype0
└
Adam/v/lstm_5/lstm_cell/biasVarHandleOp*
_output_shapes
: *-

debug_nameAdam/v/lstm_5/lstm_cell/bias/*
dtype0*
shape:ђ*-
shared_nameAdam/v/lstm_5/lstm_cell/bias
і
0Adam/v/lstm_5/lstm_cell/bias/Read/ReadVariableOpReadVariableOpAdam/v/lstm_5/lstm_cell/bias*
_output_shapes	
:ђ*
dtype0
└
Adam/m/lstm_5/lstm_cell/biasVarHandleOp*
_output_shapes
: *-

debug_nameAdam/m/lstm_5/lstm_cell/bias/*
dtype0*
shape:ђ*-
shared_nameAdam/m/lstm_5/lstm_cell/bias
і
0Adam/m/lstm_5/lstm_cell/bias/Read/ReadVariableOpReadVariableOpAdam/m/lstm_5/lstm_cell/bias*
_output_shapes	
:ђ*
dtype0
У
(Adam/v/lstm_5/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *9

debug_name+)Adam/v/lstm_5/lstm_cell/recurrent_kernel/*
dtype0*
shape:	@ђ*9
shared_name*(Adam/v/lstm_5/lstm_cell/recurrent_kernel
д
<Adam/v/lstm_5/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp(Adam/v/lstm_5/lstm_cell/recurrent_kernel*
_output_shapes
:	@ђ*
dtype0
У
(Adam/m/lstm_5/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *9

debug_name+)Adam/m/lstm_5/lstm_cell/recurrent_kernel/*
dtype0*
shape:	@ђ*9
shared_name*(Adam/m/lstm_5/lstm_cell/recurrent_kernel
д
<Adam/m/lstm_5/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp(Adam/m/lstm_5/lstm_cell/recurrent_kernel*
_output_shapes
:	@ђ*
dtype0
╦
Adam/v/lstm_5/lstm_cell/kernelVarHandleOp*
_output_shapes
: */

debug_name!Adam/v/lstm_5/lstm_cell/kernel/*
dtype0*
shape:
ђђ*/
shared_name Adam/v/lstm_5/lstm_cell/kernel
Њ
2Adam/v/lstm_5/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/v/lstm_5/lstm_cell/kernel* 
_output_shapes
:
ђђ*
dtype0
╦
Adam/m/lstm_5/lstm_cell/kernelVarHandleOp*
_output_shapes
: */

debug_name!Adam/m/lstm_5/lstm_cell/kernel/*
dtype0*
shape:
ђђ*/
shared_name Adam/m/lstm_5/lstm_cell/kernel
Њ
2Adam/m/lstm_5/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/m/lstm_5/lstm_cell/kernel* 
_output_shapes
:
ђђ*
dtype0
╚
Adam/v/embedding_5/embeddingsVarHandleOp*
_output_shapes
: *.

debug_name Adam/v/embedding_5/embeddings/*
dtype0*
shape:
ѕ'ђ*.
shared_nameAdam/v/embedding_5/embeddings
Љ
1Adam/v/embedding_5/embeddings/Read/ReadVariableOpReadVariableOpAdam/v/embedding_5/embeddings* 
_output_shapes
:
ѕ'ђ*
dtype0
╚
Adam/m/embedding_5/embeddingsVarHandleOp*
_output_shapes
: *.

debug_name Adam/m/embedding_5/embeddings/*
dtype0*
shape:
ѕ'ђ*.
shared_nameAdam/m/embedding_5/embeddings
Љ
1Adam/m/embedding_5/embeddings/Read/ReadVariableOpReadVariableOpAdam/m/embedding_5/embeddings* 
_output_shapes
:
ѕ'ђ*
dtype0
ј
learning_rateVarHandleOp*
_output_shapes
: *

debug_namelearning_rate/*
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
ѓ
	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
Ф
lstm_5/lstm_cell/biasVarHandleOp*
_output_shapes
: *&

debug_namelstm_5/lstm_cell/bias/*
dtype0*
shape:ђ*&
shared_namelstm_5/lstm_cell/bias
|
)lstm_5/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm_5/lstm_cell/bias*
_output_shapes	
:ђ*
dtype0
М
!lstm_5/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *2

debug_name$"lstm_5/lstm_cell/recurrent_kernel/*
dtype0*
shape:	@ђ*2
shared_name#!lstm_5/lstm_cell/recurrent_kernel
ў
5lstm_5/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp!lstm_5/lstm_cell/recurrent_kernel*
_output_shapes
:	@ђ*
dtype0
Х
lstm_5/lstm_cell/kernelVarHandleOp*
_output_shapes
: *(

debug_namelstm_5/lstm_cell/kernel/*
dtype0*
shape:
ђђ*(
shared_namelstm_5/lstm_cell/kernel
Ё
+lstm_5/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm_5/lstm_cell/kernel* 
_output_shapes
:
ђђ*
dtype0
Ј
dense_5/biasVarHandleOp*
_output_shapes
: *

debug_namedense_5/bias/*
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0
Ў
dense_5/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_5/kernel/*
dtype0*
shape
:@*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:@*
dtype0
│
embedding_5/embeddingsVarHandleOp*
_output_shapes
: *'

debug_nameembedding_5/embeddings/*
dtype0*
shape:
ѕ'ђ*'
shared_nameembedding_5/embeddings
Ѓ
*embedding_5/embeddings/Read/ReadVariableOpReadVariableOpembedding_5/embeddings* 
_output_shapes
:
ѕ'ђ*
dtype0
є
!serving_default_embedding_5_inputPlaceholder*(
_output_shapes
:         У*
dtype0*
shape:         У
л
StatefulPartitionedCallStatefulPartitionedCall!serving_default_embedding_5_inputembedding_5/embeddingslstm_5/lstm_cell/kernellstm_5/lstm_cell/bias!lstm_5/lstm_cell/recurrent_kerneldense_5/kerneldense_5/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *,
f'R%
#__inference_signature_wrapper_63804

NoOpNoOp
█:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ќ:
valueї:BЅ: Bѓ:
█
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
а
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings*
┴
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec*
д
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias*
Ц
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses
-_random_generator* 
ј
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses* 
.
0
41
52
63
%4
&5*
.
0
41
52
63
%4
&5*
* 
░
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

<trace_0
=trace_1* 

>trace_0
?trace_1* 
* 
Ђ
@
_variables
A_iterations
B_learning_rate
C_index_dict
D
_momentums
E_velocities
F_update_step_xla*

Gserving_default* 

0*

0*
* 
Њ
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Mtrace_0* 

Ntrace_0* 
jd
VARIABLE_VALUEembedding_5/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

40
51
62*

40
51
62*
* 
Ъ

Ostates
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Utrace_0
Vtrace_1
Wtrace_2
Xtrace_3* 
6
Ytrace_0
Ztrace_1
[trace_2
\trace_3* 
* 
с
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses
c_random_generator
d
state_size

4kernel
5recurrent_kernel
6bias*
* 

%0
&1*

%0
&1*
* 
Њ
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

jtrace_0* 

ktrace_0* 
^X
VARIABLE_VALUEdense_5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Љ
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses* 

qtrace_0
rtrace_1* 

strace_0
ttrace_1* 
* 
* 
* 
* 
Љ
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses* 

ztrace_0* 

{trace_0* 
WQ
VARIABLE_VALUElstm_5/lstm_cell/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!lstm_5/lstm_cell/recurrent_kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUElstm_5/lstm_cell/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

|0
}1*
* 
* 
* 
* 
* 
* 
l
A0
~1
2
ђ3
Ђ4
ѓ5
Ѓ6
ё7
Ё8
є9
Є10
ѕ11
Ѕ12*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
3
~0
ђ1
ѓ2
ё3
є4
ѕ5*
3
0
Ђ1
Ѓ2
Ё3
Є4
Ѕ5*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

40
51
62*

40
51
62*
* 
ў
іnon_trainable_variables
Іlayers
їmetrics
 Їlayer_regularization_losses
јlayer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses*

Јtrace_0
љtrace_1* 

Љtrace_0
њtrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
Њ	variables
ћ	keras_api

Ћtotal

ќcount*
M
Ќ	variables
ў	keras_api

Ўtotal

џcount
Џ
_fn_kwargs*
hb
VARIABLE_VALUEAdam/m/embedding_5/embeddings1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/v/embedding_5/embeddings1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/m/lstm_5/lstm_cell/kernel1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/v/lstm_5/lstm_cell/kernel1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE(Adam/m/lstm_5/lstm_cell/recurrent_kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE(Adam/v/lstm_5/lstm_cell/recurrent_kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/m/lstm_5/lstm_cell/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/v/lstm_5/lstm_cell/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_5/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_5/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_5/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_5/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 

Ћ0
ќ1*

Њ	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Ў0
џ1*

Ќ	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
█
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameembedding_5/embeddingsdense_5/kerneldense_5/biaslstm_5/lstm_cell/kernel!lstm_5/lstm_cell/recurrent_kernellstm_5/lstm_cell/bias	iterationlearning_rateAdam/m/embedding_5/embeddingsAdam/v/embedding_5/embeddingsAdam/m/lstm_5/lstm_cell/kernelAdam/v/lstm_5/lstm_cell/kernel(Adam/m/lstm_5/lstm_cell/recurrent_kernel(Adam/v/lstm_5/lstm_cell/recurrent_kernelAdam/m/lstm_5/lstm_cell/biasAdam/v/lstm_5/lstm_cell/biasAdam/m/dense_5/kernelAdam/v/dense_5/kernelAdam/m/dense_5/biasAdam/v/dense_5/biastotal_1count_1totalcountConst*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *'
f"R 
__inference__traced_save_65544
о
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_5/embeddingsdense_5/kerneldense_5/biaslstm_5/lstm_cell/kernel!lstm_5/lstm_cell/recurrent_kernellstm_5/lstm_cell/bias	iterationlearning_rateAdam/m/embedding_5/embeddingsAdam/v/embedding_5/embeddingsAdam/m/lstm_5/lstm_cell/kernelAdam/v/lstm_5/lstm_cell/kernel(Adam/m/lstm_5/lstm_cell/recurrent_kernel(Adam/v/lstm_5/lstm_cell/recurrent_kernelAdam/m/lstm_5/lstm_cell/biasAdam/v/lstm_5/lstm_cell/biasAdam/m/dense_5/kernelAdam/v/dense_5/kernelAdam/m/dense_5/biasAdam/v/dense_5/biastotal_1count_1totalcount*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference__traced_restore_65625њП
й	
Й
while_cond_63544
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_63544___redundant_placeholder03
/while_while_cond_63544___redundant_placeholder13
/while_while_cond_63544___redundant_placeholder23
/while_while_cond_63544___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
╦
b
)__inference_dropout_5_layer_call_fn_65092

inputs
identityѕбStatefulPartitionedCall┐
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_63425o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
О
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_65114

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
А┬
╦
A__inference_lstm_5_layer_call_and_return_conditional_losses_64831

inputs;
'lstm_cell_split_readvariableop_resource:
ђђ8
)lstm_cell_split_1_readvariableop_resource:	ђ4
!lstm_cell_readvariableop_resource:	@ђ
identityѕбlstm_cell/ReadVariableOpбlstm_cell/ReadVariableOp_1бlstm_cell/ReadVariableOp_2бlstm_cell/ReadVariableOp_3бlstm_cell/split/ReadVariableOpб lstm_cell/split_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          o
	transpose	Transposeinputstranspose/perm:output:0*
T0*-
_output_shapes
:У         ђR
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ђ*
shrink_axis_maskl
lstm_cell/ones_likeOnesLikestrided_slice_2:output:0*
T0*(
_output_shapes
:         ђ\
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?і
lstm_cell/dropout/MulMullstm_cell/ones_like:y:0 lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:         ђl
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:y:0*
T0*
_output_shapes
::ь¤А
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0e
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>┼
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ^
lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ╝
lstm_cell/dropout/SelectV2SelectV2"lstm_cell/dropout/GreaterEqual:z:0lstm_cell/dropout/Mul:z:0"lstm_cell/dropout/Const_1:output:0*
T0*(
_output_shapes
:         ђ^
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?ј
lstm_cell/dropout_1/MulMullstm_cell/ones_like:y:0"lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:         ђn
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:y:0*
T0*
_output_shapes
::ь¤Ц
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0g
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>╦
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ`
lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ─
lstm_cell/dropout_1/SelectV2SelectV2$lstm_cell/dropout_1/GreaterEqual:z:0lstm_cell/dropout_1/Mul:z:0$lstm_cell/dropout_1/Const_1:output:0*
T0*(
_output_shapes
:         ђ^
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?ј
lstm_cell/dropout_2/MulMullstm_cell/ones_like:y:0"lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:         ђn
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:y:0*
T0*
_output_shapes
::ь¤Ц
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0g
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>╦
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ`
lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ─
lstm_cell/dropout_2/SelectV2SelectV2$lstm_cell/dropout_2/GreaterEqual:z:0lstm_cell/dropout_2/Mul:z:0$lstm_cell/dropout_2/Const_1:output:0*
T0*(
_output_shapes
:         ђ^
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?ј
lstm_cell/dropout_3/MulMullstm_cell/ones_like:y:0"lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:         ђn
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:y:0*
T0*
_output_shapes
::ь¤Ц
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0g
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>╦
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ`
lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ─
lstm_cell/dropout_3/SelectV2SelectV2$lstm_cell/dropout_3/GreaterEqual:z:0lstm_cell/dropout_3/Mul:z:0$lstm_cell/dropout_3/Const_1:output:0*
T0*(
_output_shapes
:         ђc
lstm_cell/ones_like_1OnesLikezeros:output:0*
T0*'
_output_shapes
:         @^
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?Ј
lstm_cell/dropout_4/MulMullstm_cell/ones_like_1:y:0"lstm_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:         @p
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::ь¤ц
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0g
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>╩
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @`
lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ├
lstm_cell/dropout_4/SelectV2SelectV2$lstm_cell/dropout_4/GreaterEqual:z:0lstm_cell/dropout_4/Mul:z:0$lstm_cell/dropout_4/Const_1:output:0*
T0*'
_output_shapes
:         @^
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?Ј
lstm_cell/dropout_5/MulMullstm_cell/ones_like_1:y:0"lstm_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:         @p
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::ь¤ц
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0g
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>╩
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @`
lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ├
lstm_cell/dropout_5/SelectV2SelectV2$lstm_cell/dropout_5/GreaterEqual:z:0lstm_cell/dropout_5/Mul:z:0$lstm_cell/dropout_5/Const_1:output:0*
T0*'
_output_shapes
:         @^
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?Ј
lstm_cell/dropout_6/MulMullstm_cell/ones_like_1:y:0"lstm_cell/dropout_6/Const:output:0*
T0*'
_output_shapes
:         @p
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::ь¤ц
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0g
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>╩
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @`
lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ├
lstm_cell/dropout_6/SelectV2SelectV2$lstm_cell/dropout_6/GreaterEqual:z:0lstm_cell/dropout_6/Mul:z:0$lstm_cell/dropout_6/Const_1:output:0*
T0*'
_output_shapes
:         @^
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?Ј
lstm_cell/dropout_7/MulMullstm_cell/ones_like_1:y:0"lstm_cell/dropout_7/Const:output:0*
T0*'
_output_shapes
:         @p
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::ь¤ц
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0g
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>╩
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @`
lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ├
lstm_cell/dropout_7/SelectV2SelectV2$lstm_cell/dropout_7/GreaterEqual:z:0lstm_cell/dropout_7/Mul:z:0$lstm_cell/dropout_7/Const_1:output:0*
T0*'
_output_shapes
:         @є
lstm_cell/mulMulstrided_slice_2:output:0#lstm_cell/dropout/SelectV2:output:0*
T0*(
_output_shapes
:         ђі
lstm_cell/mul_1Mulstrided_slice_2:output:0%lstm_cell/dropout_1/SelectV2:output:0*
T0*(
_output_shapes
:         ђі
lstm_cell/mul_2Mulstrided_slice_2:output:0%lstm_cell/dropout_2/SelectV2:output:0*
T0*(
_output_shapes
:         ђі
lstm_cell/mul_3Mulstrided_slice_2:output:0%lstm_cell/dropout_3/SelectV2:output:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ѕ
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0└
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	ђ@:	ђ@:	ђ@:	ђ@*
	num_splity
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*'
_output_shapes
:         @}
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*'
_output_shapes
:         @}
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*'
_output_shapes
:         @}
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*'
_output_shapes
:         @]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Є
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype0▓
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitє
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:         @і
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:         @і
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:         @і
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:         @
lstm_cell/mul_4Mulzeros:output:0%lstm_cell/dropout_4/SelectV2:output:0*
T0*'
_output_shapes
:         @
lstm_cell/mul_5Mulzeros:output:0%lstm_cell/dropout_5/SelectV2:output:0*
T0*'
_output_shapes
:         @
lstm_cell/mul_6Mulzeros:output:0%lstm_cell/dropout_6/SelectV2:output:0*
T0*'
_output_shapes
:         @
lstm_cell/mul_7Mulzeros:output:0%lstm_cell/dropout_7/SelectV2:output:0*
T0*'
_output_shapes
:         @{
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ю
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЁ
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:         @ѓ
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:         @a
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:         @}
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ђ   r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Д
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЄ
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:         @є
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:         @e
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:         @s
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         @}
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ђ   r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Д
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЄ
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:         @є
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:         @]
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*'
_output_shapes
:         @s
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:         @t
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:         @}
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Д
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЄ
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:         @є
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:         @e
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:         @_
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*'
_output_shapes
:         @x
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : л
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_64636*
condR
while_cond_64635*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         @Я
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         Уђ: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:         Уђ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Щ
ц
F__inference_embedding_5_layer_call_and_return_conditional_losses_63820

inputs*
embedding_lookup_63815:
ѕ'ђ
identityѕбembedding_lookupV
CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:         У╗
embedding_lookupResourceGatherembedding_lookup_63815Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/63815*-
_output_shapes
:         Уђ*
dtype0x
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*-
_output_shapes
:         Уђw
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*-
_output_shapes
:         Уђ5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:         У: 2$
embedding_lookupembedding_lookup:P L
(
_output_shapes
:         У
 
_user_specified_nameinputs:%!

_user_specified_name63815
▒
Ђ
+__inference_embedding_5_layer_call_fn_63811

inputs
unknown:
ѕ'ђ
identityѕбStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         Уђ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_embedding_5_layer_call_and_return_conditional_losses_63023u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:         Уђ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:         У: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         У
 
_user_specified_nameinputs:%!

_user_specified_name63807
ю
Ъ
G__inference_sequential_5_layer_call_and_return_conditional_losses_63697
embedding_5_input%
embedding_5_63437:
ѕ'ђ 
lstm_5_63677:
ђђ
lstm_5_63679:	ђ
lstm_5_63681:	@ђ
dense_5_63684:@
dense_5_63686:
identityѕбdense_5/StatefulPartitionedCallб#embedding_5/StatefulPartitionedCallбlstm_5/StatefulPartitionedCallш
#embedding_5/StatefulPartitionedCallStatefulPartitionedCallembedding_5_inputembedding_5_63437*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         Уђ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_embedding_5_layer_call_and_return_conditional_losses_63023Џ
lstm_5/StatefulPartitionedCallStatefulPartitionedCall,embedding_5/StatefulPartitionedCall:output:0lstm_5_63677lstm_5_63679lstm_5_63681*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_63676і
dense_5/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0dense_5_63684dense_5_63686*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_63408█
dropout_5/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_63693█
activation_5/PartitionedCallPartitionedCall"dropout_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_63431t
IdentityIdentity%activation_5/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         І
NoOpNoOp ^dense_5/StatefulPartitionedCall$^embedding_5/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         У: : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2J
#embedding_5/StatefulPartitionedCall#embedding_5/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall:[ W
(
_output_shapes
:         У
+
_user_specified_nameembedding_5_input:%!

_user_specified_name63437:%!

_user_specified_name63677:%!

_user_specified_name63679:%!

_user_specified_name63681:%!

_user_specified_name63684:%!

_user_specified_name63686
ў

c
D__inference_dropout_5_layer_call_and_return_conditional_losses_65109

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ь¤ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Њ
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:         a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
й	
Й
while_cond_63195
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_63195___redundant_placeholder03
/while_while_cond_63195___redundant_placeholder13
/while_while_cond_63195___redundant_placeholder23
/while_while_cond_63195___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
▀
Х
&__inference_lstm_5_layer_call_fn_63842
inputs_0
unknown:
ђђ
	unknown_0:	ђ
	unknown_1:	@ђ
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_62956o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ђ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:                  ђ
"
_user_specified_name
inputs_0:%!

_user_specified_name63834:%!

_user_specified_name63836:%!

_user_specified_name63838
с

џ
,__inference_sequential_5_layer_call_fn_63731
embedding_5_input
unknown:
ѕ'ђ
	unknown_0:
ђђ
	unknown_1:	ђ
	unknown_2:	@ђ
	unknown_3:@
	unknown_4:
identityѕбStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallembedding_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_63697o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         У: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
(
_output_shapes
:         У
+
_user_specified_nameembedding_5_input:%!

_user_specified_name63717:%!

_user_specified_name63719:%!

_user_specified_name63721:%!

_user_specified_name63723:%!

_user_specified_name63725:%!

_user_specified_name63727
й	
Й
while_cond_64033
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_64033___redundant_placeholder03
/while_while_cond_64033___redundant_placeholder13
/while_while_cond_64033___redundant_placeholder23
/while_while_cond_64033___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
ф
╬
$sequential_5_lstm_5_while_cond_62366D
@sequential_5_lstm_5_while_sequential_5_lstm_5_while_loop_counterJ
Fsequential_5_lstm_5_while_sequential_5_lstm_5_while_maximum_iterations)
%sequential_5_lstm_5_while_placeholder+
'sequential_5_lstm_5_while_placeholder_1+
'sequential_5_lstm_5_while_placeholder_2+
'sequential_5_lstm_5_while_placeholder_3F
Bsequential_5_lstm_5_while_less_sequential_5_lstm_5_strided_slice_1[
Wsequential_5_lstm_5_while_sequential_5_lstm_5_while_cond_62366___redundant_placeholder0[
Wsequential_5_lstm_5_while_sequential_5_lstm_5_while_cond_62366___redundant_placeholder1[
Wsequential_5_lstm_5_while_sequential_5_lstm_5_while_cond_62366___redundant_placeholder2[
Wsequential_5_lstm_5_while_sequential_5_lstm_5_while_cond_62366___redundant_placeholder3&
"sequential_5_lstm_5_while_identity
▓
sequential_5/lstm_5/while/LessLess%sequential_5_lstm_5_while_placeholderBsequential_5_lstm_5_while_less_sequential_5_lstm_5_strided_slice_1*
T0*
_output_shapes
: s
"sequential_5/lstm_5/while/IdentityIdentity"sequential_5/lstm_5/while/Less:z:0*
T0
*
_output_shapes
: "Q
"sequential_5_lstm_5_while_identity+sequential_5/lstm_5/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: :::::^ Z

_output_shapes
: 
@
_user_specified_name(&sequential_5/lstm_5/while/loop_counter:d`

_output_shapes
: 
F
_user_specified_name.,sequential_5/lstm_5/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:[W

_output_shapes
: 
=
_user_specified_name%#sequential_5/lstm_5/strided_slice_1:

_output_shapes
:
¤p
Ђ	
while_body_63545
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
/while_lstm_cell_split_readvariableop_resource_0:
ђђ@
1while_lstm_cell_split_1_readvariableop_resource_0:	ђ<
)while_lstm_cell_readvariableop_resource_0:	@ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
-while_lstm_cell_split_readvariableop_resource:
ђђ>
/while_lstm_cell_split_1_readvariableop_resource:	ђ:
'while_lstm_cell_readvariableop_resource:	@ђѕбwhile/lstm_cell/ReadVariableOpб while/lstm_cell/ReadVariableOp_1б while/lstm_cell/ReadVariableOp_2б while/lstm_cell/ReadVariableOp_3б$while/lstm_cell/split/ReadVariableOpб&while/lstm_cell/split_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   Д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ђ*
element_dtype0і
while/lstm_cell/ones_likeOnesLike0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*(
_output_shapes
:         ђn
while/lstm_cell/ones_like_1OnesLikewhile_placeholder_2*
T0*'
_output_shapes
:         @ъ
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђа
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђа
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђа
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ќ
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0м
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	ђ@:	ђ@:	ђ@:	ђ@*
	num_splitІ
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:         @Ј
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:         @Ј
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:         @Ј
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:         @c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ћ
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0─
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitў
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:         @ю
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:         @ю
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:         @ю
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:         @ё
while/lstm_cell/mul_4Mulwhile_placeholder_2while/lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @ё
while/lstm_cell/mul_5Mulwhile_placeholder_2while/lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @ё
while/lstm_cell/mul_6Mulwhile_placeholder_2while/lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @ё
while/lstm_cell/mul_7Mulwhile_placeholder_2while/lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @Ѕ
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@ђ*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╗
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЌ
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:         @ћ
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:         @m
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:         @І
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@ђ*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ђ   x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┼
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЎ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:         @ў
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:         @q
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         @ѓ
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @І
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@ђ*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ђ   x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┼
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЎ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:         @ў
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:         @i
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:         @Ё
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:         @є
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:         @І
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@ђ*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┼
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЎ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:         @ў
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:         @q
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:         @k
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:         @і
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:         @r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:         @v
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:         @ѓ

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
╔A
е
D__inference_lstm_cell_layer_call_and_return_conditional_losses_65378

inputs
states_0
states_11
split_readvariableop_resource:
ђђ.
split_1_readvariableop_resource:	ђ*
readvariableop_resource:	@ђ
identity

identity_1

identity_2ѕбReadVariableOpбReadVariableOp_1бReadVariableOp_2бReadVariableOp_3бsplit/ReadVariableOpбsplit_1/ReadVariableOpP
	ones_likeOnesLikeinputs*
T0*(
_output_shapes
:         ђS
ones_like_1OnesLikestates_0*
T0*'
_output_shapes
:         @T
mulMulinputsones_like:y:0*
T0*(
_output_shapes
:         ђV
mul_1Mulinputsones_like:y:0*
T0*(
_output_shapes
:         ђV
mul_2Mulinputsones_like:y:0*
T0*(
_output_shapes
:         ђV
mul_3Mulinputsones_like:y:0*
T0*(
_output_shapes
:         ђQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0б
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	ђ@:	ђ@:	ђ@:	ђ@*
	num_split[
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:         @_
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:         @_
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:         @_
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:         @S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ћ
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:         @l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:         @l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:         @l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:         @Y
mul_4Mulstates_0ones_like_1:y:0*
T0*'
_output_shapes
:         @Y
mul_5Mulstates_0ones_like_1:y:0*
T0*'
_output_shapes
:         @Y
mul_6Mulstates_0ones_like_1:y:0*
T0*'
_output_shapes
:         @Y
mul_7Mulstates_0ones_like_1:y:0*
T0*'
_output_shapes
:         @g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	@ђ*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      в
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskg
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:         @d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:         @M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         @i
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	@ђ*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ђ   h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ш
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:         @h
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:         @Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         @W
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         @i
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	@ђ*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ђ   h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ш
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:         @h
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:         @I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:         @U
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @V
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:         @i
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	@ђ*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ш
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:         @h
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:         @Q
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:         @Z
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @Y
IdentityIdentity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:         @[

Identity_1Identity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:         @Z

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:         @ю
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         ђ:         @:         @: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs:QM
'
_output_shapes
:         @
"
_user_specified_name
states_0:QM
'
_output_shapes
:         @
"
_user_specified_name
states_1:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
┴┬
═
A__inference_lstm_5_layer_call_and_return_conditional_losses_64229
inputs_0;
'lstm_cell_split_readvariableop_resource:
ђђ8
)lstm_cell_split_1_readvariableop_resource:	ђ4
!lstm_cell_readvariableop_resource:	@ђ
identityѕбlstm_cell/ReadVariableOpбlstm_cell/ReadVariableOp_1бlstm_cell/ReadVariableOp_2бlstm_cell/ReadVariableOp_3бlstm_cell/split/ReadVariableOpб lstm_cell/split_1/ReadVariableOpбwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:                  ђR
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ђ*
shrink_axis_maskl
lstm_cell/ones_likeOnesLikestrided_slice_2:output:0*
T0*(
_output_shapes
:         ђ\
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?і
lstm_cell/dropout/MulMullstm_cell/ones_like:y:0 lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:         ђl
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:y:0*
T0*
_output_shapes
::ь¤А
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0e
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>┼
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ^
lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ╝
lstm_cell/dropout/SelectV2SelectV2"lstm_cell/dropout/GreaterEqual:z:0lstm_cell/dropout/Mul:z:0"lstm_cell/dropout/Const_1:output:0*
T0*(
_output_shapes
:         ђ^
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?ј
lstm_cell/dropout_1/MulMullstm_cell/ones_like:y:0"lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:         ђn
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:y:0*
T0*
_output_shapes
::ь¤Ц
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0g
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>╦
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ`
lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ─
lstm_cell/dropout_1/SelectV2SelectV2$lstm_cell/dropout_1/GreaterEqual:z:0lstm_cell/dropout_1/Mul:z:0$lstm_cell/dropout_1/Const_1:output:0*
T0*(
_output_shapes
:         ђ^
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?ј
lstm_cell/dropout_2/MulMullstm_cell/ones_like:y:0"lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:         ђn
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:y:0*
T0*
_output_shapes
::ь¤Ц
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0g
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>╦
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ`
lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ─
lstm_cell/dropout_2/SelectV2SelectV2$lstm_cell/dropout_2/GreaterEqual:z:0lstm_cell/dropout_2/Mul:z:0$lstm_cell/dropout_2/Const_1:output:0*
T0*(
_output_shapes
:         ђ^
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?ј
lstm_cell/dropout_3/MulMullstm_cell/ones_like:y:0"lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:         ђn
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:y:0*
T0*
_output_shapes
::ь¤Ц
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0g
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>╦
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ`
lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ─
lstm_cell/dropout_3/SelectV2SelectV2$lstm_cell/dropout_3/GreaterEqual:z:0lstm_cell/dropout_3/Mul:z:0$lstm_cell/dropout_3/Const_1:output:0*
T0*(
_output_shapes
:         ђc
lstm_cell/ones_like_1OnesLikezeros:output:0*
T0*'
_output_shapes
:         @^
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?Ј
lstm_cell/dropout_4/MulMullstm_cell/ones_like_1:y:0"lstm_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:         @p
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::ь¤ц
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0g
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>╩
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @`
lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ├
lstm_cell/dropout_4/SelectV2SelectV2$lstm_cell/dropout_4/GreaterEqual:z:0lstm_cell/dropout_4/Mul:z:0$lstm_cell/dropout_4/Const_1:output:0*
T0*'
_output_shapes
:         @^
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?Ј
lstm_cell/dropout_5/MulMullstm_cell/ones_like_1:y:0"lstm_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:         @p
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::ь¤ц
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0g
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>╩
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @`
lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ├
lstm_cell/dropout_5/SelectV2SelectV2$lstm_cell/dropout_5/GreaterEqual:z:0lstm_cell/dropout_5/Mul:z:0$lstm_cell/dropout_5/Const_1:output:0*
T0*'
_output_shapes
:         @^
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?Ј
lstm_cell/dropout_6/MulMullstm_cell/ones_like_1:y:0"lstm_cell/dropout_6/Const:output:0*
T0*'
_output_shapes
:         @p
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::ь¤ц
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0g
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>╩
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @`
lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ├
lstm_cell/dropout_6/SelectV2SelectV2$lstm_cell/dropout_6/GreaterEqual:z:0lstm_cell/dropout_6/Mul:z:0$lstm_cell/dropout_6/Const_1:output:0*
T0*'
_output_shapes
:         @^
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?Ј
lstm_cell/dropout_7/MulMullstm_cell/ones_like_1:y:0"lstm_cell/dropout_7/Const:output:0*
T0*'
_output_shapes
:         @p
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::ь¤ц
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0g
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>╩
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @`
lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ├
lstm_cell/dropout_7/SelectV2SelectV2$lstm_cell/dropout_7/GreaterEqual:z:0lstm_cell/dropout_7/Mul:z:0$lstm_cell/dropout_7/Const_1:output:0*
T0*'
_output_shapes
:         @є
lstm_cell/mulMulstrided_slice_2:output:0#lstm_cell/dropout/SelectV2:output:0*
T0*(
_output_shapes
:         ђі
lstm_cell/mul_1Mulstrided_slice_2:output:0%lstm_cell/dropout_1/SelectV2:output:0*
T0*(
_output_shapes
:         ђі
lstm_cell/mul_2Mulstrided_slice_2:output:0%lstm_cell/dropout_2/SelectV2:output:0*
T0*(
_output_shapes
:         ђі
lstm_cell/mul_3Mulstrided_slice_2:output:0%lstm_cell/dropout_3/SelectV2:output:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ѕ
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0└
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	ђ@:	ђ@:	ђ@:	ђ@*
	num_splity
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*'
_output_shapes
:         @}
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*'
_output_shapes
:         @}
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*'
_output_shapes
:         @}
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*'
_output_shapes
:         @]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Є
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype0▓
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitє
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:         @і
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:         @і
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:         @і
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:         @
lstm_cell/mul_4Mulzeros:output:0%lstm_cell/dropout_4/SelectV2:output:0*
T0*'
_output_shapes
:         @
lstm_cell/mul_5Mulzeros:output:0%lstm_cell/dropout_5/SelectV2:output:0*
T0*'
_output_shapes
:         @
lstm_cell/mul_6Mulzeros:output:0%lstm_cell/dropout_6/SelectV2:output:0*
T0*'
_output_shapes
:         @
lstm_cell/mul_7Mulzeros:output:0%lstm_cell/dropout_7/SelectV2:output:0*
T0*'
_output_shapes
:         @{
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ю
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЁ
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:         @ѓ
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:         @a
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:         @}
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ђ   r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Д
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЄ
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:         @є
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:         @e
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:         @s
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         @}
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ђ   r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Д
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЄ
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:         @є
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:         @]
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*'
_output_shapes
:         @s
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:         @t
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:         @}
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Д
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЄ
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:         @є
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:         @e
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:         @_
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*'
_output_shapes
:         @x
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : л
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_64034*
condR
while_cond_64033*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         @Я
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ђ: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:                  ђ
"
_user_specified_name
inputs_0:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
▀
Х
&__inference_lstm_5_layer_call_fn_63831
inputs_0
unknown:
ђђ
	unknown_0:	ђ
	unknown_1:	@ђ
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_62763o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ђ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:                  ђ
"
_user_specified_name
inputs_0:%!

_user_specified_name63823:%!

_user_specified_name63825:%!

_user_specified_name63827
ы}
д
D__inference_lstm_cell_layer_call_and_return_conditional_losses_62678

inputs

states
states_11
split_readvariableop_resource:
ђђ.
split_1_readvariableop_resource:	ђ*
readvariableop_resource:	@ђ
identity

identity_1

identity_2ѕбReadVariableOpбReadVariableOp_1бReadVariableOp_2бReadVariableOp_3бsplit/ReadVariableOpбsplit_1/ReadVariableOpP
	ones_likeOnesLikeinputs*
T0*(
_output_shapes
:         ђR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?l
dropout/MulMulones_like:y:0dropout/Const:output:0*
T0*(
_output_shapes
:         ђX
dropout/ShapeShapeones_like:y:0*
T0*
_output_shapes
::ь¤Ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ћ
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         ђT
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?p
dropout_1/MulMulones_like:y:0dropout_1/Const:output:0*
T0*(
_output_shapes
:         ђZ
dropout_1/ShapeShapeones_like:y:0*
T0*
_output_shapes
::ь¤Љ
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Г
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђV
dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ю
dropout_1/SelectV2SelectV2dropout_1/GreaterEqual:z:0dropout_1/Mul:z:0dropout_1/Const_1:output:0*
T0*(
_output_shapes
:         ђT
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?p
dropout_2/MulMulones_like:y:0dropout_2/Const:output:0*
T0*(
_output_shapes
:         ђZ
dropout_2/ShapeShapeones_like:y:0*
T0*
_output_shapes
::ь¤Љ
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Г
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђV
dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ю
dropout_2/SelectV2SelectV2dropout_2/GreaterEqual:z:0dropout_2/Mul:z:0dropout_2/Const_1:output:0*
T0*(
_output_shapes
:         ђT
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?p
dropout_3/MulMulones_like:y:0dropout_3/Const:output:0*
T0*(
_output_shapes
:         ђZ
dropout_3/ShapeShapeones_like:y:0*
T0*
_output_shapes
::ь¤Љ
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Г
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђV
dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ю
dropout_3/SelectV2SelectV2dropout_3/GreaterEqual:z:0dropout_3/Mul:z:0dropout_3/Const_1:output:0*
T0*(
_output_shapes
:         ђQ
ones_like_1OnesLikestates*
T0*'
_output_shapes
:         @T
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?q
dropout_4/MulMulones_like_1:y:0dropout_4/Const:output:0*
T0*'
_output_shapes
:         @\
dropout_4/ShapeShapeones_like_1:y:0*
T0*
_output_shapes
::ь¤љ
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0]
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>г
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @V
dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Џ
dropout_4/SelectV2SelectV2dropout_4/GreaterEqual:z:0dropout_4/Mul:z:0dropout_4/Const_1:output:0*
T0*'
_output_shapes
:         @T
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?q
dropout_5/MulMulones_like_1:y:0dropout_5/Const:output:0*
T0*'
_output_shapes
:         @\
dropout_5/ShapeShapeones_like_1:y:0*
T0*
_output_shapes
::ь¤љ
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0]
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>г
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @V
dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Џ
dropout_5/SelectV2SelectV2dropout_5/GreaterEqual:z:0dropout_5/Mul:z:0dropout_5/Const_1:output:0*
T0*'
_output_shapes
:         @T
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?q
dropout_6/MulMulones_like_1:y:0dropout_6/Const:output:0*
T0*'
_output_shapes
:         @\
dropout_6/ShapeShapeones_like_1:y:0*
T0*
_output_shapes
::ь¤љ
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0]
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>г
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @V
dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Џ
dropout_6/SelectV2SelectV2dropout_6/GreaterEqual:z:0dropout_6/Mul:z:0dropout_6/Const_1:output:0*
T0*'
_output_shapes
:         @T
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?q
dropout_7/MulMulones_like_1:y:0dropout_7/Const:output:0*
T0*'
_output_shapes
:         @\
dropout_7/ShapeShapeones_like_1:y:0*
T0*
_output_shapes
::ь¤љ
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0]
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>г
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @V
dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Џ
dropout_7/SelectV2SelectV2dropout_7/GreaterEqual:z:0dropout_7/Mul:z:0dropout_7/Const_1:output:0*
T0*'
_output_shapes
:         @`
mulMulinputsdropout/SelectV2:output:0*
T0*(
_output_shapes
:         ђd
mul_1Mulinputsdropout_1/SelectV2:output:0*
T0*(
_output_shapes
:         ђd
mul_2Mulinputsdropout_2/SelectV2:output:0*
T0*(
_output_shapes
:         ђd
mul_3Mulinputsdropout_3/SelectV2:output:0*
T0*(
_output_shapes
:         ђQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0б
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	ђ@:	ђ@:	ђ@:	ђ@*
	num_split[
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:         @_
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:         @_
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:         @_
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:         @S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ћ
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:         @l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:         @l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:         @l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:         @c
mul_4Mulstatesdropout_4/SelectV2:output:0*
T0*'
_output_shapes
:         @c
mul_5Mulstatesdropout_5/SelectV2:output:0*
T0*'
_output_shapes
:         @c
mul_6Mulstatesdropout_6/SelectV2:output:0*
T0*'
_output_shapes
:         @c
mul_7Mulstatesdropout_7/SelectV2:output:0*
T0*'
_output_shapes
:         @g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	@ђ*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      в
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskg
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:         @d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:         @M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         @i
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	@ђ*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ђ   h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ш
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:         @h
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:         @Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         @W
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         @i
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	@ђ*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ђ   h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ш
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:         @h
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:         @I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:         @U
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @V
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:         @i
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	@ђ*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ш
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:         @h
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:         @Q
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:         @Z
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @Y
IdentityIdentity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:         @[

Identity_1Identity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:         @Z

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:         @ю
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         ђ:         @:         @: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_namestates:OK
'
_output_shapes
:         @
 
_user_specified_namestates:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
с

џ
,__inference_sequential_5_layer_call_fn_63714
embedding_5_input
unknown:
ѕ'ђ
	unknown_0:
ђђ
	unknown_1:	ђ
	unknown_2:	@ђ
	unknown_3:@
	unknown_4:
identityѕбStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallembedding_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_63434o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         У: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
(
_output_shapes
:         У
+
_user_specified_nameembedding_5_input:%!

_user_specified_name63700:%!

_user_specified_name63702:%!

_user_specified_name63704:%!

_user_specified_name63706:%!

_user_specified_name63708:%!

_user_specified_name63710
═%
╔
while_body_62886
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_lstm_cell_62910_0:
ђђ&
while_lstm_cell_62912_0:	ђ*
while_lstm_cell_62914_0:	@ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_lstm_cell_62910:
ђђ$
while_lstm_cell_62912:	ђ(
while_lstm_cell_62914:	@ђѕб'while/lstm_cell/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   Д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ђ*
element_dtype0а
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_62910_0while_lstm_cell_62912_0while_lstm_cell_62914_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         @:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_62871r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ђ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:00while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Ї
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         @Ї
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         @R

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_lstm_cell_62910while_lstm_cell_62910_0"0
while_lstm_cell_62912while_lstm_cell_62912_0"0
while_lstm_cell_62914while_lstm_cell_62914_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:%!

_user_specified_name62910:%	!

_user_specified_name62912:%
!

_user_specified_name62914
ЈЙ
Ђ	
while_body_64636
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
/while_lstm_cell_split_readvariableop_resource_0:
ђђ@
1while_lstm_cell_split_1_readvariableop_resource_0:	ђ<
)while_lstm_cell_readvariableop_resource_0:	@ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
-while_lstm_cell_split_readvariableop_resource:
ђђ>
/while_lstm_cell_split_1_readvariableop_resource:	ђ:
'while_lstm_cell_readvariableop_resource:	@ђѕбwhile/lstm_cell/ReadVariableOpб while/lstm_cell/ReadVariableOp_1б while/lstm_cell/ReadVariableOp_2б while/lstm_cell/ReadVariableOp_3б$while/lstm_cell/split/ReadVariableOpб&while/lstm_cell/split_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   Д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ђ*
element_dtype0і
while/lstm_cell/ones_likeOnesLike0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*(
_output_shapes
:         ђb
while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?ю
while/lstm_cell/dropout/MulMulwhile/lstm_cell/ones_like:y:0&while/lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:         ђx
while/lstm_cell/dropout/ShapeShapewhile/lstm_cell/ones_like:y:0*
T0*
_output_shapes
::ь¤Г
4while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform&while/lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0k
&while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>О
$while/lstm_cell/dropout/GreaterEqualGreaterEqual=while/lstm_cell/dropout/random_uniform/RandomUniform:output:0/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђd
while/lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    н
 while/lstm_cell/dropout/SelectV2SelectV2(while/lstm_cell/dropout/GreaterEqual:z:0while/lstm_cell/dropout/Mul:z:0(while/lstm_cell/dropout/Const_1:output:0*
T0*(
_output_shapes
:         ђd
while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?а
while/lstm_cell/dropout_1/MulMulwhile/lstm_cell/ones_like:y:0(while/lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:         ђz
while/lstm_cell/dropout_1/ShapeShapewhile/lstm_cell/ones_like:y:0*
T0*
_output_shapes
::ь¤▒
6while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0m
(while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>П
&while/lstm_cell/dropout_1/GreaterEqualGreaterEqual?while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђf
!while/lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ▄
"while/lstm_cell/dropout_1/SelectV2SelectV2*while/lstm_cell/dropout_1/GreaterEqual:z:0!while/lstm_cell/dropout_1/Mul:z:0*while/lstm_cell/dropout_1/Const_1:output:0*
T0*(
_output_shapes
:         ђd
while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?а
while/lstm_cell/dropout_2/MulMulwhile/lstm_cell/ones_like:y:0(while/lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:         ђz
while/lstm_cell/dropout_2/ShapeShapewhile/lstm_cell/ones_like:y:0*
T0*
_output_shapes
::ь¤▒
6while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0m
(while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>П
&while/lstm_cell/dropout_2/GreaterEqualGreaterEqual?while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђf
!while/lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ▄
"while/lstm_cell/dropout_2/SelectV2SelectV2*while/lstm_cell/dropout_2/GreaterEqual:z:0!while/lstm_cell/dropout_2/Mul:z:0*while/lstm_cell/dropout_2/Const_1:output:0*
T0*(
_output_shapes
:         ђd
while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?а
while/lstm_cell/dropout_3/MulMulwhile/lstm_cell/ones_like:y:0(while/lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:         ђz
while/lstm_cell/dropout_3/ShapeShapewhile/lstm_cell/ones_like:y:0*
T0*
_output_shapes
::ь¤▒
6while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0m
(while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>П
&while/lstm_cell/dropout_3/GreaterEqualGreaterEqual?while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђf
!while/lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ▄
"while/lstm_cell/dropout_3/SelectV2SelectV2*while/lstm_cell/dropout_3/GreaterEqual:z:0!while/lstm_cell/dropout_3/Mul:z:0*while/lstm_cell/dropout_3/Const_1:output:0*
T0*(
_output_shapes
:         ђn
while/lstm_cell/ones_like_1OnesLikewhile_placeholder_2*
T0*'
_output_shapes
:         @d
while/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?А
while/lstm_cell/dropout_4/MulMulwhile/lstm_cell/ones_like_1:y:0(while/lstm_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:         @|
while/lstm_cell/dropout_4/ShapeShapewhile/lstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::ь¤░
6while/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0m
(while/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>▄
&while/lstm_cell/dropout_4/GreaterEqualGreaterEqual?while/lstm_cell/dropout_4/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @f
!while/lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    █
"while/lstm_cell/dropout_4/SelectV2SelectV2*while/lstm_cell/dropout_4/GreaterEqual:z:0!while/lstm_cell/dropout_4/Mul:z:0*while/lstm_cell/dropout_4/Const_1:output:0*
T0*'
_output_shapes
:         @d
while/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?А
while/lstm_cell/dropout_5/MulMulwhile/lstm_cell/ones_like_1:y:0(while/lstm_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:         @|
while/lstm_cell/dropout_5/ShapeShapewhile/lstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::ь¤░
6while/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0m
(while/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>▄
&while/lstm_cell/dropout_5/GreaterEqualGreaterEqual?while/lstm_cell/dropout_5/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @f
!while/lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    █
"while/lstm_cell/dropout_5/SelectV2SelectV2*while/lstm_cell/dropout_5/GreaterEqual:z:0!while/lstm_cell/dropout_5/Mul:z:0*while/lstm_cell/dropout_5/Const_1:output:0*
T0*'
_output_shapes
:         @d
while/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?А
while/lstm_cell/dropout_6/MulMulwhile/lstm_cell/ones_like_1:y:0(while/lstm_cell/dropout_6/Const:output:0*
T0*'
_output_shapes
:         @|
while/lstm_cell/dropout_6/ShapeShapewhile/lstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::ь¤░
6while/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_6/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0m
(while/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>▄
&while/lstm_cell/dropout_6/GreaterEqualGreaterEqual?while/lstm_cell/dropout_6/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @f
!while/lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    █
"while/lstm_cell/dropout_6/SelectV2SelectV2*while/lstm_cell/dropout_6/GreaterEqual:z:0!while/lstm_cell/dropout_6/Mul:z:0*while/lstm_cell/dropout_6/Const_1:output:0*
T0*'
_output_shapes
:         @d
while/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?А
while/lstm_cell/dropout_7/MulMulwhile/lstm_cell/ones_like_1:y:0(while/lstm_cell/dropout_7/Const:output:0*
T0*'
_output_shapes
:         @|
while/lstm_cell/dropout_7/ShapeShapewhile/lstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::ь¤░
6while/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_7/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0m
(while/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>▄
&while/lstm_cell/dropout_7/GreaterEqualGreaterEqual?while/lstm_cell/dropout_7/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @f
!while/lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    █
"while/lstm_cell/dropout_7/SelectV2SelectV2*while/lstm_cell/dropout_7/GreaterEqual:z:0!while/lstm_cell/dropout_7/Mul:z:0*while/lstm_cell/dropout_7/Const_1:output:0*
T0*'
_output_shapes
:         @ф
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/lstm_cell/dropout/SelectV2:output:0*
T0*(
_output_shapes
:         ђ«
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_1/SelectV2:output:0*
T0*(
_output_shapes
:         ђ«
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_2/SelectV2:output:0*
T0*(
_output_shapes
:         ђ«
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_3/SelectV2:output:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ќ
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0м
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	ђ@:	ђ@:	ђ@:	ђ@*
	num_splitІ
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:         @Ј
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:         @Ј
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:         @Ј
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:         @c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ћ
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0─
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitў
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:         @ю
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:         @ю
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:         @ю
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:         @љ
while/lstm_cell/mul_4Mulwhile_placeholder_2+while/lstm_cell/dropout_4/SelectV2:output:0*
T0*'
_output_shapes
:         @љ
while/lstm_cell/mul_5Mulwhile_placeholder_2+while/lstm_cell/dropout_5/SelectV2:output:0*
T0*'
_output_shapes
:         @љ
while/lstm_cell/mul_6Mulwhile_placeholder_2+while/lstm_cell/dropout_6/SelectV2:output:0*
T0*'
_output_shapes
:         @љ
while/lstm_cell/mul_7Mulwhile_placeholder_2+while/lstm_cell/dropout_7/SelectV2:output:0*
T0*'
_output_shapes
:         @Ѕ
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@ђ*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╗
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЌ
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:         @ћ
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:         @m
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:         @І
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@ђ*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ђ   x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┼
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЎ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:         @ў
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:         @q
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         @ѓ
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @І
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@ђ*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ђ   x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┼
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЎ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:         @ў
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:         @i
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:         @Ё
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:         @є
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:         @І
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@ђ*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┼
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЎ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:         @ў
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:         @q
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:         @k
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:         @і
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:         @r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:         @v
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:         @ѓ

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
Ю
E
)__inference_dropout_5_layer_call_fn_65097

inputs
identity»
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_63693`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ђ~
е
D__inference_lstm_cell_layer_call_and_return_conditional_losses_65300

inputs
states_0
states_11
split_readvariableop_resource:
ђђ.
split_1_readvariableop_resource:	ђ*
readvariableop_resource:	@ђ
identity

identity_1

identity_2ѕбReadVariableOpбReadVariableOp_1бReadVariableOp_2бReadVariableOp_3бsplit/ReadVariableOpбsplit_1/ReadVariableOpP
	ones_likeOnesLikeinputs*
T0*(
_output_shapes
:         ђR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?l
dropout/MulMulones_like:y:0dropout/Const:output:0*
T0*(
_output_shapes
:         ђX
dropout/ShapeShapeones_like:y:0*
T0*
_output_shapes
::ь¤Ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ћ
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         ђT
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?p
dropout_1/MulMulones_like:y:0dropout_1/Const:output:0*
T0*(
_output_shapes
:         ђZ
dropout_1/ShapeShapeones_like:y:0*
T0*
_output_shapes
::ь¤Љ
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Г
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђV
dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ю
dropout_1/SelectV2SelectV2dropout_1/GreaterEqual:z:0dropout_1/Mul:z:0dropout_1/Const_1:output:0*
T0*(
_output_shapes
:         ђT
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?p
dropout_2/MulMulones_like:y:0dropout_2/Const:output:0*
T0*(
_output_shapes
:         ђZ
dropout_2/ShapeShapeones_like:y:0*
T0*
_output_shapes
::ь¤Љ
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Г
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђV
dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ю
dropout_2/SelectV2SelectV2dropout_2/GreaterEqual:z:0dropout_2/Mul:z:0dropout_2/Const_1:output:0*
T0*(
_output_shapes
:         ђT
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?p
dropout_3/MulMulones_like:y:0dropout_3/Const:output:0*
T0*(
_output_shapes
:         ђZ
dropout_3/ShapeShapeones_like:y:0*
T0*
_output_shapes
::ь¤Љ
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Г
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђV
dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ю
dropout_3/SelectV2SelectV2dropout_3/GreaterEqual:z:0dropout_3/Mul:z:0dropout_3/Const_1:output:0*
T0*(
_output_shapes
:         ђS
ones_like_1OnesLikestates_0*
T0*'
_output_shapes
:         @T
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?q
dropout_4/MulMulones_like_1:y:0dropout_4/Const:output:0*
T0*'
_output_shapes
:         @\
dropout_4/ShapeShapeones_like_1:y:0*
T0*
_output_shapes
::ь¤љ
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0]
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>г
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @V
dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Џ
dropout_4/SelectV2SelectV2dropout_4/GreaterEqual:z:0dropout_4/Mul:z:0dropout_4/Const_1:output:0*
T0*'
_output_shapes
:         @T
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?q
dropout_5/MulMulones_like_1:y:0dropout_5/Const:output:0*
T0*'
_output_shapes
:         @\
dropout_5/ShapeShapeones_like_1:y:0*
T0*
_output_shapes
::ь¤љ
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0]
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>г
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @V
dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Џ
dropout_5/SelectV2SelectV2dropout_5/GreaterEqual:z:0dropout_5/Mul:z:0dropout_5/Const_1:output:0*
T0*'
_output_shapes
:         @T
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?q
dropout_6/MulMulones_like_1:y:0dropout_6/Const:output:0*
T0*'
_output_shapes
:         @\
dropout_6/ShapeShapeones_like_1:y:0*
T0*
_output_shapes
::ь¤љ
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0]
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>г
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @V
dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Џ
dropout_6/SelectV2SelectV2dropout_6/GreaterEqual:z:0dropout_6/Mul:z:0dropout_6/Const_1:output:0*
T0*'
_output_shapes
:         @T
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?q
dropout_7/MulMulones_like_1:y:0dropout_7/Const:output:0*
T0*'
_output_shapes
:         @\
dropout_7/ShapeShapeones_like_1:y:0*
T0*
_output_shapes
::ь¤љ
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0]
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>г
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @V
dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Џ
dropout_7/SelectV2SelectV2dropout_7/GreaterEqual:z:0dropout_7/Mul:z:0dropout_7/Const_1:output:0*
T0*'
_output_shapes
:         @`
mulMulinputsdropout/SelectV2:output:0*
T0*(
_output_shapes
:         ђd
mul_1Mulinputsdropout_1/SelectV2:output:0*
T0*(
_output_shapes
:         ђd
mul_2Mulinputsdropout_2/SelectV2:output:0*
T0*(
_output_shapes
:         ђd
mul_3Mulinputsdropout_3/SelectV2:output:0*
T0*(
_output_shapes
:         ђQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0б
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	ђ@:	ђ@:	ђ@:	ђ@*
	num_split[
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:         @_
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:         @_
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:         @_
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:         @S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ћ
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:         @l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:         @l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:         @l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:         @e
mul_4Mulstates_0dropout_4/SelectV2:output:0*
T0*'
_output_shapes
:         @e
mul_5Mulstates_0dropout_5/SelectV2:output:0*
T0*'
_output_shapes
:         @e
mul_6Mulstates_0dropout_6/SelectV2:output:0*
T0*'
_output_shapes
:         @e
mul_7Mulstates_0dropout_7/SelectV2:output:0*
T0*'
_output_shapes
:         @g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	@ђ*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      в
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskg
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:         @d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:         @M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         @i
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	@ђ*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ђ   h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ш
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:         @h
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:         @Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         @W
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         @i
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	@ђ*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ђ   h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ш
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:         @h
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:         @I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:         @U
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @V
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:         @i
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	@ђ*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ш
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:         @h
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:         @Q
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:         @Z
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @Y
IdentityIdentity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:         @[

Identity_1Identity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:         @Z

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:         @ю
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         ђ:         @:         @: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs:QM
'
_output_shapes
:         @
"
_user_specified_name
states_0:QM
'
_output_shapes
:         @
"
_user_specified_name
states_1:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
╩
c
G__inference_activation_5_layer_call_and_return_conditional_losses_65124

inputs
identityL
SigmoidSigmoidinputs*
T0*'
_output_shapes
:         S
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Щ
ц
F__inference_embedding_5_layer_call_and_return_conditional_losses_63023

inputs*
embedding_lookup_63018:
ѕ'ђ
identityѕбembedding_lookupV
CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:         У╗
embedding_lookupResourceGatherembedding_lookup_63018Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/63018*-
_output_shapes
:         Уђ*
dtype0x
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*-
_output_shapes
:         Уђw
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*-
_output_shapes
:         Уђ5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:         У: 2$
embedding_lookupembedding_lookup:P L
(
_output_shapes
:         У
 
_user_specified_nameinputs:%!

_user_specified_name63018
Ќ9
ш
A__inference_lstm_5_layer_call_and_return_conditional_losses_62763

inputs#
lstm_cell_62679:
ђђ
lstm_cell_62681:	ђ"
lstm_cell_62683:	@ђ
identityѕб!lstm_cell/StatefulPartitionedCallбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:                  ђR
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ђ*
shrink_axis_maskР
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_62679lstm_cell_62681lstm_cell_62683*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         @:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_62678n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ї
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_62679lstm_cell_62681lstm_cell_62683*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_62693*
condR
while_cond_62692*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         @N
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ђ: : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:                  ђ
 
_user_specified_nameinputs:%!

_user_specified_name62679:%!

_user_specified_name62681:%!

_user_specified_name62683
Б
H
,__inference_activation_5_layer_call_fn_65119

inputs
identity▓
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_63431`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ЈЙ
Ђ	
while_body_64034
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
/while_lstm_cell_split_readvariableop_resource_0:
ђђ@
1while_lstm_cell_split_1_readvariableop_resource_0:	ђ<
)while_lstm_cell_readvariableop_resource_0:	@ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
-while_lstm_cell_split_readvariableop_resource:
ђђ>
/while_lstm_cell_split_1_readvariableop_resource:	ђ:
'while_lstm_cell_readvariableop_resource:	@ђѕбwhile/lstm_cell/ReadVariableOpб while/lstm_cell/ReadVariableOp_1б while/lstm_cell/ReadVariableOp_2б while/lstm_cell/ReadVariableOp_3б$while/lstm_cell/split/ReadVariableOpб&while/lstm_cell/split_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   Д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ђ*
element_dtype0і
while/lstm_cell/ones_likeOnesLike0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*(
_output_shapes
:         ђb
while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?ю
while/lstm_cell/dropout/MulMulwhile/lstm_cell/ones_like:y:0&while/lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:         ђx
while/lstm_cell/dropout/ShapeShapewhile/lstm_cell/ones_like:y:0*
T0*
_output_shapes
::ь¤Г
4while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform&while/lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0k
&while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>О
$while/lstm_cell/dropout/GreaterEqualGreaterEqual=while/lstm_cell/dropout/random_uniform/RandomUniform:output:0/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђd
while/lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    н
 while/lstm_cell/dropout/SelectV2SelectV2(while/lstm_cell/dropout/GreaterEqual:z:0while/lstm_cell/dropout/Mul:z:0(while/lstm_cell/dropout/Const_1:output:0*
T0*(
_output_shapes
:         ђd
while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?а
while/lstm_cell/dropout_1/MulMulwhile/lstm_cell/ones_like:y:0(while/lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:         ђz
while/lstm_cell/dropout_1/ShapeShapewhile/lstm_cell/ones_like:y:0*
T0*
_output_shapes
::ь¤▒
6while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0m
(while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>П
&while/lstm_cell/dropout_1/GreaterEqualGreaterEqual?while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђf
!while/lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ▄
"while/lstm_cell/dropout_1/SelectV2SelectV2*while/lstm_cell/dropout_1/GreaterEqual:z:0!while/lstm_cell/dropout_1/Mul:z:0*while/lstm_cell/dropout_1/Const_1:output:0*
T0*(
_output_shapes
:         ђd
while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?а
while/lstm_cell/dropout_2/MulMulwhile/lstm_cell/ones_like:y:0(while/lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:         ђz
while/lstm_cell/dropout_2/ShapeShapewhile/lstm_cell/ones_like:y:0*
T0*
_output_shapes
::ь¤▒
6while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0m
(while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>П
&while/lstm_cell/dropout_2/GreaterEqualGreaterEqual?while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђf
!while/lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ▄
"while/lstm_cell/dropout_2/SelectV2SelectV2*while/lstm_cell/dropout_2/GreaterEqual:z:0!while/lstm_cell/dropout_2/Mul:z:0*while/lstm_cell/dropout_2/Const_1:output:0*
T0*(
_output_shapes
:         ђd
while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?а
while/lstm_cell/dropout_3/MulMulwhile/lstm_cell/ones_like:y:0(while/lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:         ђz
while/lstm_cell/dropout_3/ShapeShapewhile/lstm_cell/ones_like:y:0*
T0*
_output_shapes
::ь¤▒
6while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0m
(while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>П
&while/lstm_cell/dropout_3/GreaterEqualGreaterEqual?while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђf
!while/lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ▄
"while/lstm_cell/dropout_3/SelectV2SelectV2*while/lstm_cell/dropout_3/GreaterEqual:z:0!while/lstm_cell/dropout_3/Mul:z:0*while/lstm_cell/dropout_3/Const_1:output:0*
T0*(
_output_shapes
:         ђn
while/lstm_cell/ones_like_1OnesLikewhile_placeholder_2*
T0*'
_output_shapes
:         @d
while/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?А
while/lstm_cell/dropout_4/MulMulwhile/lstm_cell/ones_like_1:y:0(while/lstm_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:         @|
while/lstm_cell/dropout_4/ShapeShapewhile/lstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::ь¤░
6while/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0m
(while/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>▄
&while/lstm_cell/dropout_4/GreaterEqualGreaterEqual?while/lstm_cell/dropout_4/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @f
!while/lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    █
"while/lstm_cell/dropout_4/SelectV2SelectV2*while/lstm_cell/dropout_4/GreaterEqual:z:0!while/lstm_cell/dropout_4/Mul:z:0*while/lstm_cell/dropout_4/Const_1:output:0*
T0*'
_output_shapes
:         @d
while/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?А
while/lstm_cell/dropout_5/MulMulwhile/lstm_cell/ones_like_1:y:0(while/lstm_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:         @|
while/lstm_cell/dropout_5/ShapeShapewhile/lstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::ь¤░
6while/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0m
(while/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>▄
&while/lstm_cell/dropout_5/GreaterEqualGreaterEqual?while/lstm_cell/dropout_5/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @f
!while/lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    █
"while/lstm_cell/dropout_5/SelectV2SelectV2*while/lstm_cell/dropout_5/GreaterEqual:z:0!while/lstm_cell/dropout_5/Mul:z:0*while/lstm_cell/dropout_5/Const_1:output:0*
T0*'
_output_shapes
:         @d
while/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?А
while/lstm_cell/dropout_6/MulMulwhile/lstm_cell/ones_like_1:y:0(while/lstm_cell/dropout_6/Const:output:0*
T0*'
_output_shapes
:         @|
while/lstm_cell/dropout_6/ShapeShapewhile/lstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::ь¤░
6while/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_6/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0m
(while/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>▄
&while/lstm_cell/dropout_6/GreaterEqualGreaterEqual?while/lstm_cell/dropout_6/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @f
!while/lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    █
"while/lstm_cell/dropout_6/SelectV2SelectV2*while/lstm_cell/dropout_6/GreaterEqual:z:0!while/lstm_cell/dropout_6/Mul:z:0*while/lstm_cell/dropout_6/Const_1:output:0*
T0*'
_output_shapes
:         @d
while/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?А
while/lstm_cell/dropout_7/MulMulwhile/lstm_cell/ones_like_1:y:0(while/lstm_cell/dropout_7/Const:output:0*
T0*'
_output_shapes
:         @|
while/lstm_cell/dropout_7/ShapeShapewhile/lstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::ь¤░
6while/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_7/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0m
(while/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>▄
&while/lstm_cell/dropout_7/GreaterEqualGreaterEqual?while/lstm_cell/dropout_7/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @f
!while/lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    █
"while/lstm_cell/dropout_7/SelectV2SelectV2*while/lstm_cell/dropout_7/GreaterEqual:z:0!while/lstm_cell/dropout_7/Mul:z:0*while/lstm_cell/dropout_7/Const_1:output:0*
T0*'
_output_shapes
:         @ф
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/lstm_cell/dropout/SelectV2:output:0*
T0*(
_output_shapes
:         ђ«
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_1/SelectV2:output:0*
T0*(
_output_shapes
:         ђ«
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_2/SelectV2:output:0*
T0*(
_output_shapes
:         ђ«
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_3/SelectV2:output:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ќ
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0м
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	ђ@:	ђ@:	ђ@:	ђ@*
	num_splitІ
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:         @Ј
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:         @Ј
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:         @Ј
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:         @c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ћ
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0─
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitў
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:         @ю
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:         @ю
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:         @ю
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:         @љ
while/lstm_cell/mul_4Mulwhile_placeholder_2+while/lstm_cell/dropout_4/SelectV2:output:0*
T0*'
_output_shapes
:         @љ
while/lstm_cell/mul_5Mulwhile_placeholder_2+while/lstm_cell/dropout_5/SelectV2:output:0*
T0*'
_output_shapes
:         @љ
while/lstm_cell/mul_6Mulwhile_placeholder_2+while/lstm_cell/dropout_6/SelectV2:output:0*
T0*'
_output_shapes
:         @љ
while/lstm_cell/mul_7Mulwhile_placeholder_2+while/lstm_cell/dropout_7/SelectV2:output:0*
T0*'
_output_shapes
:         @Ѕ
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@ђ*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╗
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЌ
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:         @ћ
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:         @m
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:         @І
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@ђ*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ђ   x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┼
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЎ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:         @ў
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:         @q
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         @ѓ
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @І
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@ђ*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ђ   x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┼
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЎ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:         @ў
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:         @i
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:         @Ё
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:         @є
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:         @І
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@ђ*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┼
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЎ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:         @ў
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:         @q
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:         @k
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:         @і
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:         @r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:         @v
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:         @ѓ

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
й	
Й
while_cond_62692
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_62692___redundant_placeholder03
/while_while_cond_62692___redundant_placeholder13
/while_while_cond_62692___redundant_placeholder23
/while_while_cond_62692___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
У
ћ
'__inference_dense_5_layer_call_fn_65077

inputs
unknown:@
	unknown_0:
identityѕбStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_63408o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:%!

_user_specified_name65071:%!

_user_specified_name65073
╣A
д
D__inference_lstm_cell_layer_call_and_return_conditional_losses_62871

inputs

states
states_11
split_readvariableop_resource:
ђђ.
split_1_readvariableop_resource:	ђ*
readvariableop_resource:	@ђ
identity

identity_1

identity_2ѕбReadVariableOpбReadVariableOp_1бReadVariableOp_2бReadVariableOp_3бsplit/ReadVariableOpбsplit_1/ReadVariableOpP
	ones_likeOnesLikeinputs*
T0*(
_output_shapes
:         ђQ
ones_like_1OnesLikestates*
T0*'
_output_shapes
:         @T
mulMulinputsones_like:y:0*
T0*(
_output_shapes
:         ђV
mul_1Mulinputsones_like:y:0*
T0*(
_output_shapes
:         ђV
mul_2Mulinputsones_like:y:0*
T0*(
_output_shapes
:         ђV
mul_3Mulinputsones_like:y:0*
T0*(
_output_shapes
:         ђQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0б
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	ђ@:	ђ@:	ђ@:	ђ@*
	num_split[
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:         @_
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:         @_
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:         @_
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:         @S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ћ
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:         @l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:         @l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:         @l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:         @W
mul_4Mulstatesones_like_1:y:0*
T0*'
_output_shapes
:         @W
mul_5Mulstatesones_like_1:y:0*
T0*'
_output_shapes
:         @W
mul_6Mulstatesones_like_1:y:0*
T0*'
_output_shapes
:         @W
mul_7Mulstatesones_like_1:y:0*
T0*'
_output_shapes
:         @g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	@ђ*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      в
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskg
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:         @d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:         @M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         @i
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	@ђ*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ђ   h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ш
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:         @h
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:         @Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         @W
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         @i
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	@ђ*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ђ   h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ш
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:         @h
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:         @I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:         @U
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @V
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:         @i
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	@ђ*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ш
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:         @h
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:         @Q
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:         @Z
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @Y
IdentityIdentity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:         @[

Identity_1Identity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:         @Z

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:         @ю
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         ђ:         @:         @: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_namestates:OK
'
_output_shapes
:         @
 
_user_specified_namestates:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
М┐
М
__inference__traced_save_65544
file_prefixA
-read_disablecopyonread_embedding_5_embeddings:
ѕ'ђ9
'read_1_disablecopyonread_dense_5_kernel:@3
%read_2_disablecopyonread_dense_5_bias:D
0read_3_disablecopyonread_lstm_5_lstm_cell_kernel:
ђђM
:read_4_disablecopyonread_lstm_5_lstm_cell_recurrent_kernel:	@ђ=
.read_5_disablecopyonread_lstm_5_lstm_cell_bias:	ђ,
"read_6_disablecopyonread_iteration:	 0
&read_7_disablecopyonread_learning_rate: J
6read_8_disablecopyonread_adam_m_embedding_5_embeddings:
ѕ'ђJ
6read_9_disablecopyonread_adam_v_embedding_5_embeddings:
ѕ'ђL
8read_10_disablecopyonread_adam_m_lstm_5_lstm_cell_kernel:
ђђL
8read_11_disablecopyonread_adam_v_lstm_5_lstm_cell_kernel:
ђђU
Bread_12_disablecopyonread_adam_m_lstm_5_lstm_cell_recurrent_kernel:	@ђU
Bread_13_disablecopyonread_adam_v_lstm_5_lstm_cell_recurrent_kernel:	@ђE
6read_14_disablecopyonread_adam_m_lstm_5_lstm_cell_bias:	ђE
6read_15_disablecopyonread_adam_v_lstm_5_lstm_cell_bias:	ђA
/read_16_disablecopyonread_adam_m_dense_5_kernel:@A
/read_17_disablecopyonread_adam_v_dense_5_kernel:@;
-read_18_disablecopyonread_adam_m_dense_5_bias:;
-read_19_disablecopyonread_adam_v_dense_5_bias:+
!read_20_disablecopyonread_total_1: +
!read_21_disablecopyonread_count_1: )
read_22_disablecopyonread_total: )
read_23_disablecopyonread_count: 
savev2_const
identity_49ѕбMergeV2CheckpointsбRead/DisableCopyOnReadбRead/ReadVariableOpбRead_1/DisableCopyOnReadбRead_1/ReadVariableOpбRead_10/DisableCopyOnReadбRead_10/ReadVariableOpбRead_11/DisableCopyOnReadбRead_11/ReadVariableOpбRead_12/DisableCopyOnReadбRead_12/ReadVariableOpбRead_13/DisableCopyOnReadбRead_13/ReadVariableOpбRead_14/DisableCopyOnReadбRead_14/ReadVariableOpбRead_15/DisableCopyOnReadбRead_15/ReadVariableOpбRead_16/DisableCopyOnReadбRead_16/ReadVariableOpбRead_17/DisableCopyOnReadбRead_17/ReadVariableOpбRead_18/DisableCopyOnReadбRead_18/ReadVariableOpбRead_19/DisableCopyOnReadбRead_19/ReadVariableOpбRead_2/DisableCopyOnReadбRead_2/ReadVariableOpбRead_20/DisableCopyOnReadбRead_20/ReadVariableOpбRead_21/DisableCopyOnReadбRead_21/ReadVariableOpбRead_22/DisableCopyOnReadбRead_22/ReadVariableOpбRead_23/DisableCopyOnReadбRead_23/ReadVariableOpбRead_3/DisableCopyOnReadбRead_3/ReadVariableOpбRead_4/DisableCopyOnReadбRead_4/ReadVariableOpбRead_5/DisableCopyOnReadбRead_5/ReadVariableOpбRead_6/DisableCopyOnReadбRead_6/ReadVariableOpбRead_7/DisableCopyOnReadбRead_7/ReadVariableOpбRead_8/DisableCopyOnReadбRead_8/ReadVariableOpбRead_9/DisableCopyOnReadбRead_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partЂ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
Read/DisableCopyOnReadDisableCopyOnRead-read_disablecopyonread_embedding_5_embeddings"/device:CPU:0*
_output_shapes
 Ф
Read/ReadVariableOpReadVariableOp-read_disablecopyonread_embedding_5_embeddings^Read/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
ѕ'ђ*
dtype0k
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ѕ'ђc

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ѕ'ђ{
Read_1/DisableCopyOnReadDisableCopyOnRead'read_1_disablecopyonread_dense_5_kernel"/device:CPU:0*
_output_shapes
 Д
Read_1/ReadVariableOpReadVariableOp'read_1_disablecopyonread_dense_5_kernel^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0m

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@c

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes

:@y
Read_2/DisableCopyOnReadDisableCopyOnRead%read_2_disablecopyonread_dense_5_bias"/device:CPU:0*
_output_shapes
 А
Read_2/ReadVariableOpReadVariableOp%read_2_disablecopyonread_dense_5_bias^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:ё
Read_3/DisableCopyOnReadDisableCopyOnRead0read_3_disablecopyonread_lstm_5_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 ▓
Read_3/ReadVariableOpReadVariableOp0read_3_disablecopyonread_lstm_5_lstm_cell_kernel^Read_3/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
ђђ*
dtype0o

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ђђe

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ђђј
Read_4/DisableCopyOnReadDisableCopyOnRead:read_4_disablecopyonread_lstm_5_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 ╗
Read_4/ReadVariableOpReadVariableOp:read_4_disablecopyonread_lstm_5_lstm_cell_recurrent_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	@ђ*
dtype0n

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	@ђd

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:	@ђѓ
Read_5/DisableCopyOnReadDisableCopyOnRead.read_5_disablecopyonread_lstm_5_lstm_cell_bias"/device:CPU:0*
_output_shapes
 Ф
Read_5/ReadVariableOpReadVariableOp.read_5_disablecopyonread_lstm_5_lstm_cell_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђv
Read_6/DisableCopyOnReadDisableCopyOnRead"read_6_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 џ
Read_6/ReadVariableOpReadVariableOp"read_6_disablecopyonread_iteration^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 ъ
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_learning_rate^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: і
Read_8/DisableCopyOnReadDisableCopyOnRead6read_8_disablecopyonread_adam_m_embedding_5_embeddings"/device:CPU:0*
_output_shapes
 И
Read_8/ReadVariableOpReadVariableOp6read_8_disablecopyonread_adam_m_embedding_5_embeddings^Read_8/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
ѕ'ђ*
dtype0p
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ѕ'ђg
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ѕ'ђі
Read_9/DisableCopyOnReadDisableCopyOnRead6read_9_disablecopyonread_adam_v_embedding_5_embeddings"/device:CPU:0*
_output_shapes
 И
Read_9/ReadVariableOpReadVariableOp6read_9_disablecopyonread_adam_v_embedding_5_embeddings^Read_9/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
ѕ'ђ*
dtype0p
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ѕ'ђg
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ѕ'ђЇ
Read_10/DisableCopyOnReadDisableCopyOnRead8read_10_disablecopyonread_adam_m_lstm_5_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_10/ReadVariableOpReadVariableOp8read_10_disablecopyonread_adam_m_lstm_5_lstm_cell_kernel^Read_10/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
ђђ*
dtype0q
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ђђg
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ђђЇ
Read_11/DisableCopyOnReadDisableCopyOnRead8read_11_disablecopyonread_adam_v_lstm_5_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_11/ReadVariableOpReadVariableOp8read_11_disablecopyonread_adam_v_lstm_5_lstm_cell_kernel^Read_11/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
ђђ*
dtype0q
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ђђg
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ђђЌ
Read_12/DisableCopyOnReadDisableCopyOnReadBread_12_disablecopyonread_adam_m_lstm_5_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 ┼
Read_12/ReadVariableOpReadVariableOpBread_12_disablecopyonread_adam_m_lstm_5_lstm_cell_recurrent_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	@ђ*
dtype0p
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	@ђf
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:	@ђЌ
Read_13/DisableCopyOnReadDisableCopyOnReadBread_13_disablecopyonread_adam_v_lstm_5_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 ┼
Read_13/ReadVariableOpReadVariableOpBread_13_disablecopyonread_adam_v_lstm_5_lstm_cell_recurrent_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	@ђ*
dtype0p
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	@ђf
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:	@ђІ
Read_14/DisableCopyOnReadDisableCopyOnRead6read_14_disablecopyonread_adam_m_lstm_5_lstm_cell_bias"/device:CPU:0*
_output_shapes
 х
Read_14/ReadVariableOpReadVariableOp6read_14_disablecopyonread_adam_m_lstm_5_lstm_cell_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђІ
Read_15/DisableCopyOnReadDisableCopyOnRead6read_15_disablecopyonread_adam_v_lstm_5_lstm_cell_bias"/device:CPU:0*
_output_shapes
 х
Read_15/ReadVariableOpReadVariableOp6read_15_disablecopyonread_adam_v_lstm_5_lstm_cell_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђё
Read_16/DisableCopyOnReadDisableCopyOnRead/read_16_disablecopyonread_adam_m_dense_5_kernel"/device:CPU:0*
_output_shapes
 ▒
Read_16/ReadVariableOpReadVariableOp/read_16_disablecopyonread_adam_m_dense_5_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

:@ё
Read_17/DisableCopyOnReadDisableCopyOnRead/read_17_disablecopyonread_adam_v_dense_5_kernel"/device:CPU:0*
_output_shapes
 ▒
Read_17/ReadVariableOpReadVariableOp/read_17_disablecopyonread_adam_v_dense_5_kernel^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes

:@ѓ
Read_18/DisableCopyOnReadDisableCopyOnRead-read_18_disablecopyonread_adam_m_dense_5_bias"/device:CPU:0*
_output_shapes
 Ф
Read_18/ReadVariableOpReadVariableOp-read_18_disablecopyonread_adam_m_dense_5_bias^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:ѓ
Read_19/DisableCopyOnReadDisableCopyOnRead-read_19_disablecopyonread_adam_v_dense_5_bias"/device:CPU:0*
_output_shapes
 Ф
Read_19/ReadVariableOpReadVariableOp-read_19_disablecopyonread_adam_v_dense_5_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_20/DisableCopyOnReadDisableCopyOnRead!read_20_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 Џ
Read_20/ReadVariableOpReadVariableOp!read_20_disablecopyonread_total_1^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_21/DisableCopyOnReadDisableCopyOnRead!read_21_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 Џ
Read_21/ReadVariableOpReadVariableOp!read_21_disablecopyonread_count_1^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_22/DisableCopyOnReadDisableCopyOnReadread_22_disablecopyonread_total"/device:CPU:0*
_output_shapes
 Ў
Read_22/ReadVariableOpReadVariableOpread_22_disablecopyonread_total^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_23/DisableCopyOnReadDisableCopyOnReadread_23_disablecopyonread_count"/device:CPU:0*
_output_shapes
 Ў
Read_23/ReadVariableOpReadVariableOpread_23_disablecopyonread_count^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: м

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ч	
valueы	BЬ	B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЪ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B ч
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *'
dtypes
2	љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_48Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_49IdentityIdentity_48:output:0^NoOp*
T0*
_output_shapes
: Ќ

NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_49Identity_49:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:62
0
_user_specified_nameembedding_5/embeddings:.*
(
_user_specified_namedense_5/kernel:,(
&
_user_specified_namedense_5/bias:73
1
_user_specified_namelstm_5/lstm_cell/kernel:A=
;
_user_specified_name#!lstm_5/lstm_cell/recurrent_kernel:51
/
_user_specified_namelstm_5/lstm_cell/bias:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namelearning_rate:=	9
7
_user_specified_nameAdam/m/embedding_5/embeddings:=
9
7
_user_specified_nameAdam/v/embedding_5/embeddings:>:
8
_user_specified_name Adam/m/lstm_5/lstm_cell/kernel:>:
8
_user_specified_name Adam/v/lstm_5/lstm_cell/kernel:HD
B
_user_specified_name*(Adam/m/lstm_5/lstm_cell/recurrent_kernel:HD
B
_user_specified_name*(Adam/v/lstm_5/lstm_cell/recurrent_kernel:<8
6
_user_specified_nameAdam/m/lstm_5/lstm_cell/bias:<8
6
_user_specified_nameAdam/v/lstm_5/lstm_cell/bias:51
/
_user_specified_nameAdam/m/dense_5/kernel:51
/
_user_specified_nameAdam/v/dense_5/kernel:3/
-
_user_specified_nameAdam/m/dense_5/bias:3/
-
_user_specified_nameAdam/v/dense_5/bias:'#
!
_user_specified_name	total_1:'#
!
_user_specified_name	count_1:%!

_user_specified_nametotal:%!

_user_specified_namecount:=9

_output_shapes
: 

_user_specified_nameConst
ш	
з
B__inference_dense_5_layer_call_and_return_conditional_losses_63408

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
й	
Й
while_cond_64936
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_64936___redundant_placeholder03
/while_while_cond_64936___redundant_placeholder13
/while_while_cond_64936___redundant_placeholder23
/while_while_cond_64936___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
ш	
з
B__inference_dense_5_layer_call_and_return_conditional_losses_65087

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
О
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_63693

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
¤p
Ђ	
while_body_64335
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
/while_lstm_cell_split_readvariableop_resource_0:
ђђ@
1while_lstm_cell_split_1_readvariableop_resource_0:	ђ<
)while_lstm_cell_readvariableop_resource_0:	@ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
-while_lstm_cell_split_readvariableop_resource:
ђђ>
/while_lstm_cell_split_1_readvariableop_resource:	ђ:
'while_lstm_cell_readvariableop_resource:	@ђѕбwhile/lstm_cell/ReadVariableOpб while/lstm_cell/ReadVariableOp_1б while/lstm_cell/ReadVariableOp_2б while/lstm_cell/ReadVariableOp_3б$while/lstm_cell/split/ReadVariableOpб&while/lstm_cell/split_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   Д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ђ*
element_dtype0і
while/lstm_cell/ones_likeOnesLike0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*(
_output_shapes
:         ђn
while/lstm_cell/ones_like_1OnesLikewhile_placeholder_2*
T0*'
_output_shapes
:         @ъ
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђа
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђа
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђа
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ќ
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0м
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	ђ@:	ђ@:	ђ@:	ђ@*
	num_splitІ
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:         @Ј
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:         @Ј
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:         @Ј
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:         @c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ћ
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0─
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitў
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:         @ю
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:         @ю
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:         @ю
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:         @ё
while/lstm_cell/mul_4Mulwhile_placeholder_2while/lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @ё
while/lstm_cell/mul_5Mulwhile_placeholder_2while/lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @ё
while/lstm_cell/mul_6Mulwhile_placeholder_2while/lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @ё
while/lstm_cell/mul_7Mulwhile_placeholder_2while/lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @Ѕ
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@ђ*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╗
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЌ
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:         @ћ
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:         @m
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:         @І
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@ђ*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ђ   x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┼
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЎ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:         @ў
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:         @q
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         @ѓ
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @І
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@ђ*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ђ   x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┼
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЎ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:         @ў
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:         @i
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:         @Ё
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:         @є
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:         @І
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@ђ*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┼
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЎ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:         @ў
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:         @q
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:         @k
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:         @і
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:         @r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:         @v
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:         @ѓ

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
ЈЙ
Ђ	
while_body_63196
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
/while_lstm_cell_split_readvariableop_resource_0:
ђђ@
1while_lstm_cell_split_1_readvariableop_resource_0:	ђ<
)while_lstm_cell_readvariableop_resource_0:	@ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
-while_lstm_cell_split_readvariableop_resource:
ђђ>
/while_lstm_cell_split_1_readvariableop_resource:	ђ:
'while_lstm_cell_readvariableop_resource:	@ђѕбwhile/lstm_cell/ReadVariableOpб while/lstm_cell/ReadVariableOp_1б while/lstm_cell/ReadVariableOp_2б while/lstm_cell/ReadVariableOp_3б$while/lstm_cell/split/ReadVariableOpб&while/lstm_cell/split_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   Д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ђ*
element_dtype0і
while/lstm_cell/ones_likeOnesLike0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*(
_output_shapes
:         ђb
while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?ю
while/lstm_cell/dropout/MulMulwhile/lstm_cell/ones_like:y:0&while/lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:         ђx
while/lstm_cell/dropout/ShapeShapewhile/lstm_cell/ones_like:y:0*
T0*
_output_shapes
::ь¤Г
4while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform&while/lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0k
&while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>О
$while/lstm_cell/dropout/GreaterEqualGreaterEqual=while/lstm_cell/dropout/random_uniform/RandomUniform:output:0/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђd
while/lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    н
 while/lstm_cell/dropout/SelectV2SelectV2(while/lstm_cell/dropout/GreaterEqual:z:0while/lstm_cell/dropout/Mul:z:0(while/lstm_cell/dropout/Const_1:output:0*
T0*(
_output_shapes
:         ђd
while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?а
while/lstm_cell/dropout_1/MulMulwhile/lstm_cell/ones_like:y:0(while/lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:         ђz
while/lstm_cell/dropout_1/ShapeShapewhile/lstm_cell/ones_like:y:0*
T0*
_output_shapes
::ь¤▒
6while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0m
(while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>П
&while/lstm_cell/dropout_1/GreaterEqualGreaterEqual?while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђf
!while/lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ▄
"while/lstm_cell/dropout_1/SelectV2SelectV2*while/lstm_cell/dropout_1/GreaterEqual:z:0!while/lstm_cell/dropout_1/Mul:z:0*while/lstm_cell/dropout_1/Const_1:output:0*
T0*(
_output_shapes
:         ђd
while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?а
while/lstm_cell/dropout_2/MulMulwhile/lstm_cell/ones_like:y:0(while/lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:         ђz
while/lstm_cell/dropout_2/ShapeShapewhile/lstm_cell/ones_like:y:0*
T0*
_output_shapes
::ь¤▒
6while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0m
(while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>П
&while/lstm_cell/dropout_2/GreaterEqualGreaterEqual?while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђf
!while/lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ▄
"while/lstm_cell/dropout_2/SelectV2SelectV2*while/lstm_cell/dropout_2/GreaterEqual:z:0!while/lstm_cell/dropout_2/Mul:z:0*while/lstm_cell/dropout_2/Const_1:output:0*
T0*(
_output_shapes
:         ђd
while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?а
while/lstm_cell/dropout_3/MulMulwhile/lstm_cell/ones_like:y:0(while/lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:         ђz
while/lstm_cell/dropout_3/ShapeShapewhile/lstm_cell/ones_like:y:0*
T0*
_output_shapes
::ь¤▒
6while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0m
(while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>П
&while/lstm_cell/dropout_3/GreaterEqualGreaterEqual?while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђf
!while/lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ▄
"while/lstm_cell/dropout_3/SelectV2SelectV2*while/lstm_cell/dropout_3/GreaterEqual:z:0!while/lstm_cell/dropout_3/Mul:z:0*while/lstm_cell/dropout_3/Const_1:output:0*
T0*(
_output_shapes
:         ђn
while/lstm_cell/ones_like_1OnesLikewhile_placeholder_2*
T0*'
_output_shapes
:         @d
while/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?А
while/lstm_cell/dropout_4/MulMulwhile/lstm_cell/ones_like_1:y:0(while/lstm_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:         @|
while/lstm_cell/dropout_4/ShapeShapewhile/lstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::ь¤░
6while/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0m
(while/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>▄
&while/lstm_cell/dropout_4/GreaterEqualGreaterEqual?while/lstm_cell/dropout_4/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @f
!while/lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    █
"while/lstm_cell/dropout_4/SelectV2SelectV2*while/lstm_cell/dropout_4/GreaterEqual:z:0!while/lstm_cell/dropout_4/Mul:z:0*while/lstm_cell/dropout_4/Const_1:output:0*
T0*'
_output_shapes
:         @d
while/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?А
while/lstm_cell/dropout_5/MulMulwhile/lstm_cell/ones_like_1:y:0(while/lstm_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:         @|
while/lstm_cell/dropout_5/ShapeShapewhile/lstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::ь¤░
6while/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0m
(while/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>▄
&while/lstm_cell/dropout_5/GreaterEqualGreaterEqual?while/lstm_cell/dropout_5/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @f
!while/lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    █
"while/lstm_cell/dropout_5/SelectV2SelectV2*while/lstm_cell/dropout_5/GreaterEqual:z:0!while/lstm_cell/dropout_5/Mul:z:0*while/lstm_cell/dropout_5/Const_1:output:0*
T0*'
_output_shapes
:         @d
while/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?А
while/lstm_cell/dropout_6/MulMulwhile/lstm_cell/ones_like_1:y:0(while/lstm_cell/dropout_6/Const:output:0*
T0*'
_output_shapes
:         @|
while/lstm_cell/dropout_6/ShapeShapewhile/lstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::ь¤░
6while/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_6/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0m
(while/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>▄
&while/lstm_cell/dropout_6/GreaterEqualGreaterEqual?while/lstm_cell/dropout_6/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @f
!while/lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    █
"while/lstm_cell/dropout_6/SelectV2SelectV2*while/lstm_cell/dropout_6/GreaterEqual:z:0!while/lstm_cell/dropout_6/Mul:z:0*while/lstm_cell/dropout_6/Const_1:output:0*
T0*'
_output_shapes
:         @d
while/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?А
while/lstm_cell/dropout_7/MulMulwhile/lstm_cell/ones_like_1:y:0(while/lstm_cell/dropout_7/Const:output:0*
T0*'
_output_shapes
:         @|
while/lstm_cell/dropout_7/ShapeShapewhile/lstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::ь¤░
6while/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_7/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0m
(while/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>▄
&while/lstm_cell/dropout_7/GreaterEqualGreaterEqual?while/lstm_cell/dropout_7/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @f
!while/lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    █
"while/lstm_cell/dropout_7/SelectV2SelectV2*while/lstm_cell/dropout_7/GreaterEqual:z:0!while/lstm_cell/dropout_7/Mul:z:0*while/lstm_cell/dropout_7/Const_1:output:0*
T0*'
_output_shapes
:         @ф
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/lstm_cell/dropout/SelectV2:output:0*
T0*(
_output_shapes
:         ђ«
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_1/SelectV2:output:0*
T0*(
_output_shapes
:         ђ«
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_2/SelectV2:output:0*
T0*(
_output_shapes
:         ђ«
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_3/SelectV2:output:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ќ
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0м
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	ђ@:	ђ@:	ђ@:	ђ@*
	num_splitІ
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:         @Ј
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:         @Ј
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:         @Ј
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:         @c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ћ
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0─
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitў
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:         @ю
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:         @ю
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:         @ю
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:         @љ
while/lstm_cell/mul_4Mulwhile_placeholder_2+while/lstm_cell/dropout_4/SelectV2:output:0*
T0*'
_output_shapes
:         @љ
while/lstm_cell/mul_5Mulwhile_placeholder_2+while/lstm_cell/dropout_5/SelectV2:output:0*
T0*'
_output_shapes
:         @љ
while/lstm_cell/mul_6Mulwhile_placeholder_2+while/lstm_cell/dropout_6/SelectV2:output:0*
T0*'
_output_shapes
:         @љ
while/lstm_cell/mul_7Mulwhile_placeholder_2+while/lstm_cell/dropout_7/SelectV2:output:0*
T0*'
_output_shapes
:         @Ѕ
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@ђ*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╗
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЌ
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:         @ћ
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:         @m
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:         @І
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@ђ*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ђ   x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┼
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЎ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:         @ў
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:         @q
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         @ѓ
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @І
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@ђ*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ђ   x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┼
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЎ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:         @ў
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:         @i
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:         @Ё
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:         @є
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:         @І
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@ђ*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┼
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЎ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:         @ў
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:         @q
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:         @k
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:         @і
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:         @r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:         @v
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:         @ѓ

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
│

Љ
#__inference_signature_wrapper_63804
embedding_5_input
unknown:
ѕ'ђ
	unknown_0:
ђђ
	unknown_1:	ђ
	unknown_2:	@ђ
	unknown_3:@
	unknown_4:
identityѕбStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallembedding_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *)
f$R"
 __inference__wrapped_model_62506o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         У: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
(
_output_shapes
:         У
+
_user_specified_nameembedding_5_input:%!

_user_specified_name63790:%!

_user_specified_name63792:%!

_user_specified_name63794:%!

_user_specified_name63796:%!

_user_specified_name63798:%!

_user_specified_name63800
╔
┤
&__inference_lstm_5_layer_call_fn_63853

inputs
unknown:
ђђ
	unknown_0:	ђ
	unknown_1:	@ђ
identityѕбStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_63391o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         Уђ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:         Уђ
 
_user_specified_nameinputs:%!

_user_specified_name63845:%!

_user_specified_name63847:%!

_user_specified_name63849
и
з
)__inference_lstm_cell_layer_call_fn_65141

inputs
states_0
states_1
unknown:
ђђ
	unknown_0:	ђ
	unknown_1:	@ђ
identity

identity_1

identity_2ѕбStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         @:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_62678o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         @q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         ђ:         @:         @: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs:QM
'
_output_shapes
:         @
"
_user_specified_name
states_0:QM
'
_output_shapes
:         @
"
_user_specified_name
states_1:%!

_user_specified_name65129:%!

_user_specified_name65131:%!

_user_specified_name65133
═%
╔
while_body_62693
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_lstm_cell_62717_0:
ђђ&
while_lstm_cell_62719_0:	ђ*
while_lstm_cell_62721_0:	@ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_lstm_cell_62717:
ђђ$
while_lstm_cell_62719:	ђ(
while_lstm_cell_62721:	@ђѕб'while/lstm_cell/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   Д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ђ*
element_dtype0а
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_62717_0while_lstm_cell_62719_0while_lstm_cell_62721_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         @:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_62678r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ђ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:00while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Ї
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         @Ї
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         @R

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_lstm_cell_62717while_lstm_cell_62717_0"0
while_lstm_cell_62719while_lstm_cell_62719_0"0
while_lstm_cell_62721while_lstm_cell_62721_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:%!

_user_specified_name62717:%	!

_user_specified_name62719:%
!

_user_specified_name62721
ў

c
D__inference_dropout_5_layer_call_and_return_conditional_losses_63425

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ь¤ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Њ
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:         a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ї{
╦
A__inference_lstm_5_layer_call_and_return_conditional_losses_63676

inputs;
'lstm_cell_split_readvariableop_resource:
ђђ8
)lstm_cell_split_1_readvariableop_resource:	ђ4
!lstm_cell_readvariableop_resource:	@ђ
identityѕбlstm_cell/ReadVariableOpбlstm_cell/ReadVariableOp_1бlstm_cell/ReadVariableOp_2бlstm_cell/ReadVariableOp_3бlstm_cell/split/ReadVariableOpб lstm_cell/split_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          o
	transpose	Transposeinputstranspose/perm:output:0*
T0*-
_output_shapes
:У         ђR
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ђ*
shrink_axis_maskl
lstm_cell/ones_likeOnesLikestrided_slice_2:output:0*
T0*(
_output_shapes
:         ђc
lstm_cell/ones_like_1OnesLikezeros:output:0*
T0*'
_output_shapes
:         @z
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђ|
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђ|
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђ|
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ѕ
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0└
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	ђ@:	ђ@:	ђ@:	ђ@*
	num_splity
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*'
_output_shapes
:         @}
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*'
_output_shapes
:         @}
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*'
_output_shapes
:         @}
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*'
_output_shapes
:         @]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Є
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype0▓
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitє
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:         @і
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:         @і
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:         @і
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:         @s
lstm_cell/mul_4Mulzeros:output:0lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @s
lstm_cell/mul_5Mulzeros:output:0lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @s
lstm_cell/mul_6Mulzeros:output:0lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @s
lstm_cell/mul_7Mulzeros:output:0lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @{
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ю
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЁ
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:         @ѓ
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:         @a
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:         @}
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ђ   r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Д
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЄ
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:         @є
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:         @e
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:         @s
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         @}
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ђ   r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Д
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЄ
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:         @є
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:         @]
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*'
_output_shapes
:         @s
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:         @t
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:         @}
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Д
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЄ
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:         @є
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:         @e
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:         @_
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*'
_output_shapes
:         @x
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : л
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_63545*
condR
while_cond_63544*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         @Я
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         Уђ: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:         Уђ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ќ9
ш
A__inference_lstm_5_layer_call_and_return_conditional_losses_62956

inputs#
lstm_cell_62872:
ђђ
lstm_cell_62874:	ђ"
lstm_cell_62876:	@ђ
identityѕб!lstm_cell/StatefulPartitionedCallбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:                  ђR
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ђ*
shrink_axis_maskР
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_62872lstm_cell_62874lstm_cell_62876*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         @:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_62871n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ї
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_62872lstm_cell_62874lstm_cell_62876*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_62886*
condR
while_cond_62885*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         @N
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ђ: : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:                  ђ
 
_user_specified_nameinputs:%!

_user_specified_name62872:%!

_user_specified_name62874:%!

_user_specified_name62876
Ѓ║
н
 __inference__wrapped_model_62506
embedding_5_inputC
/sequential_5_embedding_5_embedding_lookup_62260:
ѕ'ђO
;sequential_5_lstm_5_lstm_cell_split_readvariableop_resource:
ђђL
=sequential_5_lstm_5_lstm_cell_split_1_readvariableop_resource:	ђH
5sequential_5_lstm_5_lstm_cell_readvariableop_resource:	@ђE
3sequential_5_dense_5_matmul_readvariableop_resource:@B
4sequential_5_dense_5_biasadd_readvariableop_resource:
identityѕб+sequential_5/dense_5/BiasAdd/ReadVariableOpб*sequential_5/dense_5/MatMul/ReadVariableOpб)sequential_5/embedding_5/embedding_lookupб,sequential_5/lstm_5/lstm_cell/ReadVariableOpб.sequential_5/lstm_5/lstm_cell/ReadVariableOp_1б.sequential_5/lstm_5/lstm_cell/ReadVariableOp_2б.sequential_5/lstm_5/lstm_cell/ReadVariableOp_3б2sequential_5/lstm_5/lstm_cell/split/ReadVariableOpб4sequential_5/lstm_5/lstm_cell/split_1/ReadVariableOpбsequential_5/lstm_5/whilez
sequential_5/embedding_5/CastCastembedding_5_input*

DstT0*

SrcT0*(
_output_shapes
:         УЪ
)sequential_5/embedding_5/embedding_lookupResourceGather/sequential_5_embedding_5_embedding_lookup_62260!sequential_5/embedding_5/Cast:y:0*
Tindices0*B
_class8
64loc:@sequential_5/embedding_5/embedding_lookup/62260*-
_output_shapes
:         Уђ*
dtype0ф
2sequential_5/embedding_5/embedding_lookup/IdentityIdentity2sequential_5/embedding_5/embedding_lookup:output:0*
T0*-
_output_shapes
:         Уђњ
sequential_5/lstm_5/ShapeShape;sequential_5/embedding_5/embedding_lookup/Identity:output:0*
T0*
_output_shapes
::ь¤q
'sequential_5/lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_5/lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_5/lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
!sequential_5/lstm_5/strided_sliceStridedSlice"sequential_5/lstm_5/Shape:output:00sequential_5/lstm_5/strided_slice/stack:output:02sequential_5/lstm_5/strided_slice/stack_1:output:02sequential_5/lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sequential_5/lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@»
 sequential_5/lstm_5/zeros/packedPack*sequential_5/lstm_5/strided_slice:output:0+sequential_5/lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_5/lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    е
sequential_5/lstm_5/zerosFill)sequential_5/lstm_5/zeros/packed:output:0(sequential_5/lstm_5/zeros/Const:output:0*
T0*'
_output_shapes
:         @f
$sequential_5/lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@│
"sequential_5/lstm_5/zeros_1/packedPack*sequential_5/lstm_5/strided_slice:output:0-sequential_5/lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_5/lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
sequential_5/lstm_5/zeros_1Fill+sequential_5/lstm_5/zeros_1/packed:output:0*sequential_5/lstm_5/zeros_1/Const:output:0*
T0*'
_output_shapes
:         @w
"sequential_5/lstm_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ╠
sequential_5/lstm_5/transpose	Transpose;sequential_5/embedding_5/embedding_lookup/Identity:output:0+sequential_5/lstm_5/transpose/perm:output:0*
T0*-
_output_shapes
:У         ђz
sequential_5/lstm_5/Shape_1Shape!sequential_5/lstm_5/transpose:y:0*
T0*
_output_shapes
::ь¤s
)sequential_5/lstm_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_5/lstm_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_5/lstm_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#sequential_5/lstm_5/strided_slice_1StridedSlice$sequential_5/lstm_5/Shape_1:output:02sequential_5/lstm_5/strided_slice_1/stack:output:04sequential_5/lstm_5/strided_slice_1/stack_1:output:04sequential_5/lstm_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/sequential_5/lstm_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ­
!sequential_5/lstm_5/TensorArrayV2TensorListReserve8sequential_5/lstm_5/TensorArrayV2/element_shape:output:0,sequential_5/lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмџ
Isequential_5/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   ю
;sequential_5/lstm_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_5/lstm_5/transpose:y:0Rsequential_5/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмs
)sequential_5/lstm_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_5/lstm_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_5/lstm_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╬
#sequential_5/lstm_5/strided_slice_2StridedSlice!sequential_5/lstm_5/transpose:y:02sequential_5/lstm_5/strided_slice_2/stack:output:04sequential_5/lstm_5/strided_slice_2/stack_1:output:04sequential_5/lstm_5/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ђ*
shrink_axis_maskћ
'sequential_5/lstm_5/lstm_cell/ones_likeOnesLike,sequential_5/lstm_5/strided_slice_2:output:0*
T0*(
_output_shapes
:         ђІ
)sequential_5/lstm_5/lstm_cell/ones_like_1OnesLike"sequential_5/lstm_5/zeros:output:0*
T0*'
_output_shapes
:         @Х
!sequential_5/lstm_5/lstm_cell/mulMul,sequential_5/lstm_5/strided_slice_2:output:0+sequential_5/lstm_5/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђИ
#sequential_5/lstm_5/lstm_cell/mul_1Mul,sequential_5/lstm_5/strided_slice_2:output:0+sequential_5/lstm_5/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђИ
#sequential_5/lstm_5/lstm_cell/mul_2Mul,sequential_5/lstm_5/strided_slice_2:output:0+sequential_5/lstm_5/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђИ
#sequential_5/lstm_5/lstm_cell/mul_3Mul,sequential_5/lstm_5/strided_slice_2:output:0+sequential_5/lstm_5/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђo
-sequential_5/lstm_5/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :░
2sequential_5/lstm_5/lstm_cell/split/ReadVariableOpReadVariableOp;sequential_5_lstm_5_lstm_cell_split_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0Ч
#sequential_5/lstm_5/lstm_cell/splitSplit6sequential_5/lstm_5/lstm_cell/split/split_dim:output:0:sequential_5/lstm_5/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	ђ@:	ђ@:	ђ@:	ђ@*
	num_splitх
$sequential_5/lstm_5/lstm_cell/MatMulMatMul%sequential_5/lstm_5/lstm_cell/mul:z:0,sequential_5/lstm_5/lstm_cell/split:output:0*
T0*'
_output_shapes
:         @╣
&sequential_5/lstm_5/lstm_cell/MatMul_1MatMul'sequential_5/lstm_5/lstm_cell/mul_1:z:0,sequential_5/lstm_5/lstm_cell/split:output:1*
T0*'
_output_shapes
:         @╣
&sequential_5/lstm_5/lstm_cell/MatMul_2MatMul'sequential_5/lstm_5/lstm_cell/mul_2:z:0,sequential_5/lstm_5/lstm_cell/split:output:2*
T0*'
_output_shapes
:         @╣
&sequential_5/lstm_5/lstm_cell/MatMul_3MatMul'sequential_5/lstm_5/lstm_cell/mul_3:z:0,sequential_5/lstm_5/lstm_cell/split:output:3*
T0*'
_output_shapes
:         @q
/sequential_5/lstm_5/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : »
4sequential_5/lstm_5/lstm_cell/split_1/ReadVariableOpReadVariableOp=sequential_5_lstm_5_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ь
%sequential_5/lstm_5/lstm_cell/split_1Split8sequential_5/lstm_5/lstm_cell/split_1/split_dim:output:0<sequential_5/lstm_5/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split┬
%sequential_5/lstm_5/lstm_cell/BiasAddBiasAdd.sequential_5/lstm_5/lstm_cell/MatMul:product:0.sequential_5/lstm_5/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:         @к
'sequential_5/lstm_5/lstm_cell/BiasAdd_1BiasAdd0sequential_5/lstm_5/lstm_cell/MatMul_1:product:0.sequential_5/lstm_5/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:         @к
'sequential_5/lstm_5/lstm_cell/BiasAdd_2BiasAdd0sequential_5/lstm_5/lstm_cell/MatMul_2:product:0.sequential_5/lstm_5/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:         @к
'sequential_5/lstm_5/lstm_cell/BiasAdd_3BiasAdd0sequential_5/lstm_5/lstm_cell/MatMul_3:product:0.sequential_5/lstm_5/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:         @»
#sequential_5/lstm_5/lstm_cell/mul_4Mul"sequential_5/lstm_5/zeros:output:0-sequential_5/lstm_5/lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @»
#sequential_5/lstm_5/lstm_cell/mul_5Mul"sequential_5/lstm_5/zeros:output:0-sequential_5/lstm_5/lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @»
#sequential_5/lstm_5/lstm_cell/mul_6Mul"sequential_5/lstm_5/zeros:output:0-sequential_5/lstm_5/lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @»
#sequential_5/lstm_5/lstm_cell/mul_7Mul"sequential_5/lstm_5/zeros:output:0-sequential_5/lstm_5/lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @Б
,sequential_5/lstm_5/lstm_cell/ReadVariableOpReadVariableOp5sequential_5_lstm_5_lstm_cell_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0ѓ
1sequential_5/lstm_5/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ё
3sequential_5/lstm_5/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   ё
3sequential_5/lstm_5/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ђ
+sequential_5/lstm_5/lstm_cell/strided_sliceStridedSlice4sequential_5/lstm_5/lstm_cell/ReadVariableOp:value:0:sequential_5/lstm_5/lstm_cell/strided_slice/stack:output:0<sequential_5/lstm_5/lstm_cell/strided_slice/stack_1:output:0<sequential_5/lstm_5/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask┴
&sequential_5/lstm_5/lstm_cell/MatMul_4MatMul'sequential_5/lstm_5/lstm_cell/mul_4:z:04sequential_5/lstm_5/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:         @Й
!sequential_5/lstm_5/lstm_cell/addAddV2.sequential_5/lstm_5/lstm_cell/BiasAdd:output:00sequential_5/lstm_5/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:         @Ѕ
%sequential_5/lstm_5/lstm_cell/SigmoidSigmoid%sequential_5/lstm_5/lstm_cell/add:z:0*
T0*'
_output_shapes
:         @Ц
.sequential_5/lstm_5/lstm_cell/ReadVariableOp_1ReadVariableOp5sequential_5_lstm_5_lstm_cell_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0ё
3sequential_5/lstm_5/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   є
5sequential_5/lstm_5/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ђ   є
5sequential_5/lstm_5/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
-sequential_5/lstm_5/lstm_cell/strided_slice_1StridedSlice6sequential_5/lstm_5/lstm_cell/ReadVariableOp_1:value:0<sequential_5/lstm_5/lstm_cell/strided_slice_1/stack:output:0>sequential_5/lstm_5/lstm_cell/strided_slice_1/stack_1:output:0>sequential_5/lstm_5/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask├
&sequential_5/lstm_5/lstm_cell/MatMul_5MatMul'sequential_5/lstm_5/lstm_cell/mul_5:z:06sequential_5/lstm_5/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:         @┬
#sequential_5/lstm_5/lstm_cell/add_1AddV20sequential_5/lstm_5/lstm_cell/BiasAdd_1:output:00sequential_5/lstm_5/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:         @Ї
'sequential_5/lstm_5/lstm_cell/Sigmoid_1Sigmoid'sequential_5/lstm_5/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         @»
#sequential_5/lstm_5/lstm_cell/mul_8Mul+sequential_5/lstm_5/lstm_cell/Sigmoid_1:y:0$sequential_5/lstm_5/zeros_1:output:0*
T0*'
_output_shapes
:         @Ц
.sequential_5/lstm_5/lstm_cell/ReadVariableOp_2ReadVariableOp5sequential_5_lstm_5_lstm_cell_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0ё
3sequential_5/lstm_5/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ђ   є
5sequential_5/lstm_5/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   є
5sequential_5/lstm_5/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
-sequential_5/lstm_5/lstm_cell/strided_slice_2StridedSlice6sequential_5/lstm_5/lstm_cell/ReadVariableOp_2:value:0<sequential_5/lstm_5/lstm_cell/strided_slice_2/stack:output:0>sequential_5/lstm_5/lstm_cell/strided_slice_2/stack_1:output:0>sequential_5/lstm_5/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask├
&sequential_5/lstm_5/lstm_cell/MatMul_6MatMul'sequential_5/lstm_5/lstm_cell/mul_6:z:06sequential_5/lstm_5/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:         @┬
#sequential_5/lstm_5/lstm_cell/add_2AddV20sequential_5/lstm_5/lstm_cell/BiasAdd_2:output:00sequential_5/lstm_5/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:         @Ё
"sequential_5/lstm_5/lstm_cell/TanhTanh'sequential_5/lstm_5/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:         @»
#sequential_5/lstm_5/lstm_cell/mul_9Mul)sequential_5/lstm_5/lstm_cell/Sigmoid:y:0&sequential_5/lstm_5/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:         @░
#sequential_5/lstm_5/lstm_cell/add_3AddV2'sequential_5/lstm_5/lstm_cell/mul_8:z:0'sequential_5/lstm_5/lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:         @Ц
.sequential_5/lstm_5/lstm_cell/ReadVariableOp_3ReadVariableOp5sequential_5_lstm_5_lstm_cell_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0ё
3sequential_5/lstm_5/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   є
5sequential_5/lstm_5/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        є
5sequential_5/lstm_5/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
-sequential_5/lstm_5/lstm_cell/strided_slice_3StridedSlice6sequential_5/lstm_5/lstm_cell/ReadVariableOp_3:value:0<sequential_5/lstm_5/lstm_cell/strided_slice_3/stack:output:0>sequential_5/lstm_5/lstm_cell/strided_slice_3/stack_1:output:0>sequential_5/lstm_5/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask├
&sequential_5/lstm_5/lstm_cell/MatMul_7MatMul'sequential_5/lstm_5/lstm_cell/mul_7:z:06sequential_5/lstm_5/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:         @┬
#sequential_5/lstm_5/lstm_cell/add_4AddV20sequential_5/lstm_5/lstm_cell/BiasAdd_3:output:00sequential_5/lstm_5/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:         @Ї
'sequential_5/lstm_5/lstm_cell/Sigmoid_2Sigmoid'sequential_5/lstm_5/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:         @Є
$sequential_5/lstm_5/lstm_cell/Tanh_1Tanh'sequential_5/lstm_5/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:         @┤
$sequential_5/lstm_5/lstm_cell/mul_10Mul+sequential_5/lstm_5/lstm_cell/Sigmoid_2:y:0(sequential_5/lstm_5/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:         @ѓ
1sequential_5/lstm_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   r
0sequential_5/lstm_5/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ђ
#sequential_5/lstm_5/TensorArrayV2_1TensorListReserve:sequential_5/lstm_5/TensorArrayV2_1/element_shape:output:09sequential_5/lstm_5/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмZ
sequential_5/lstm_5/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,sequential_5/lstm_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         h
&sequential_5/lstm_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : У
sequential_5/lstm_5/whileWhile/sequential_5/lstm_5/while/loop_counter:output:05sequential_5/lstm_5/while/maximum_iterations:output:0!sequential_5/lstm_5/time:output:0,sequential_5/lstm_5/TensorArrayV2_1:handle:0"sequential_5/lstm_5/zeros:output:0$sequential_5/lstm_5/zeros_1:output:0,sequential_5/lstm_5/strided_slice_1:output:0Ksequential_5/lstm_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0;sequential_5_lstm_5_lstm_cell_split_readvariableop_resource=sequential_5_lstm_5_lstm_cell_split_1_readvariableop_resource5sequential_5_lstm_5_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*0
body(R&
$sequential_5_lstm_5_while_body_62367*0
cond(R&
$sequential_5_lstm_5_while_cond_62366*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations Ћ
Dsequential_5/lstm_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   њ
6sequential_5/lstm_5/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_5/lstm_5/while:output:3Msequential_5/lstm_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elements|
)sequential_5/lstm_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         u
+sequential_5/lstm_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+sequential_5/lstm_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:в
#sequential_5/lstm_5/strided_slice_3StridedSlice?sequential_5/lstm_5/TensorArrayV2Stack/TensorListStack:tensor:02sequential_5/lstm_5/strided_slice_3/stack:output:04sequential_5/lstm_5/strided_slice_3/stack_1:output:04sequential_5/lstm_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_masky
$sequential_5/lstm_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          м
sequential_5/lstm_5/transpose_1	Transpose?sequential_5/lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_5/lstm_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:         @o
sequential_5/lstm_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ъ
*sequential_5/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_5_dense_5_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0╣
sequential_5/dense_5/MatMulMatMul,sequential_5/lstm_5/strided_slice_3:output:02sequential_5/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ю
+sequential_5/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0х
sequential_5/dense_5/BiasAddBiasAdd%sequential_5/dense_5/MatMul:product:03sequential_5/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
sequential_5/dropout_5/IdentityIdentity%sequential_5/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:         ѕ
!sequential_5/activation_5/SigmoidSigmoid(sequential_5/dropout_5/Identity:output:0*
T0*'
_output_shapes
:         t
IdentityIdentity%sequential_5/activation_5/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         з
NoOpNoOp,^sequential_5/dense_5/BiasAdd/ReadVariableOp+^sequential_5/dense_5/MatMul/ReadVariableOp*^sequential_5/embedding_5/embedding_lookup-^sequential_5/lstm_5/lstm_cell/ReadVariableOp/^sequential_5/lstm_5/lstm_cell/ReadVariableOp_1/^sequential_5/lstm_5/lstm_cell/ReadVariableOp_2/^sequential_5/lstm_5/lstm_cell/ReadVariableOp_33^sequential_5/lstm_5/lstm_cell/split/ReadVariableOp5^sequential_5/lstm_5/lstm_cell/split_1/ReadVariableOp^sequential_5/lstm_5/while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         У: : : : : : 2Z
+sequential_5/dense_5/BiasAdd/ReadVariableOp+sequential_5/dense_5/BiasAdd/ReadVariableOp2X
*sequential_5/dense_5/MatMul/ReadVariableOp*sequential_5/dense_5/MatMul/ReadVariableOp2V
)sequential_5/embedding_5/embedding_lookup)sequential_5/embedding_5/embedding_lookup2\
,sequential_5/lstm_5/lstm_cell/ReadVariableOp,sequential_5/lstm_5/lstm_cell/ReadVariableOp2`
.sequential_5/lstm_5/lstm_cell/ReadVariableOp_1.sequential_5/lstm_5/lstm_cell/ReadVariableOp_12`
.sequential_5/lstm_5/lstm_cell/ReadVariableOp_2.sequential_5/lstm_5/lstm_cell/ReadVariableOp_22`
.sequential_5/lstm_5/lstm_cell/ReadVariableOp_3.sequential_5/lstm_5/lstm_cell/ReadVariableOp_32h
2sequential_5/lstm_5/lstm_cell/split/ReadVariableOp2sequential_5/lstm_5/lstm_cell/split/ReadVariableOp2l
4sequential_5/lstm_5/lstm_cell/split_1/ReadVariableOp4sequential_5/lstm_5/lstm_cell/split_1/ReadVariableOp26
sequential_5/lstm_5/whilesequential_5/lstm_5/while:[ W
(
_output_shapes
:         У
+
_user_specified_nameembedding_5_input:%!

_user_specified_name62260:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
й	
Й
while_cond_64334
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_64334___redundant_placeholder03
/while_while_cond_64334___redundant_placeholder13
/while_while_cond_64334___redundant_placeholder23
/while_while_cond_64334___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
└Џ
й
$sequential_5_lstm_5_while_body_62367D
@sequential_5_lstm_5_while_sequential_5_lstm_5_while_loop_counterJ
Fsequential_5_lstm_5_while_sequential_5_lstm_5_while_maximum_iterations)
%sequential_5_lstm_5_while_placeholder+
'sequential_5_lstm_5_while_placeholder_1+
'sequential_5_lstm_5_while_placeholder_2+
'sequential_5_lstm_5_while_placeholder_3C
?sequential_5_lstm_5_while_sequential_5_lstm_5_strided_slice_1_0
{sequential_5_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_5_tensorarrayunstack_tensorlistfromtensor_0W
Csequential_5_lstm_5_while_lstm_cell_split_readvariableop_resource_0:
ђђT
Esequential_5_lstm_5_while_lstm_cell_split_1_readvariableop_resource_0:	ђP
=sequential_5_lstm_5_while_lstm_cell_readvariableop_resource_0:	@ђ&
"sequential_5_lstm_5_while_identity(
$sequential_5_lstm_5_while_identity_1(
$sequential_5_lstm_5_while_identity_2(
$sequential_5_lstm_5_while_identity_3(
$sequential_5_lstm_5_while_identity_4(
$sequential_5_lstm_5_while_identity_5A
=sequential_5_lstm_5_while_sequential_5_lstm_5_strided_slice_1}
ysequential_5_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_5_tensorarrayunstack_tensorlistfromtensorU
Asequential_5_lstm_5_while_lstm_cell_split_readvariableop_resource:
ђђR
Csequential_5_lstm_5_while_lstm_cell_split_1_readvariableop_resource:	ђN
;sequential_5_lstm_5_while_lstm_cell_readvariableop_resource:	@ђѕб2sequential_5/lstm_5/while/lstm_cell/ReadVariableOpб4sequential_5/lstm_5/while/lstm_cell/ReadVariableOp_1б4sequential_5/lstm_5/while/lstm_cell/ReadVariableOp_2б4sequential_5/lstm_5/while/lstm_cell/ReadVariableOp_3б8sequential_5/lstm_5/while/lstm_cell/split/ReadVariableOpб:sequential_5/lstm_5/while/lstm_cell/split_1/ReadVariableOpю
Ksequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   І
=sequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_5_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_5_tensorarrayunstack_tensorlistfromtensor_0%sequential_5_lstm_5_while_placeholderTsequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ђ*
element_dtype0▓
-sequential_5/lstm_5/while/lstm_cell/ones_likeOnesLikeDsequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*(
_output_shapes
:         ђќ
/sequential_5/lstm_5/while/lstm_cell/ones_like_1OnesLike'sequential_5_lstm_5_while_placeholder_2*
T0*'
_output_shapes
:         @┌
'sequential_5/lstm_5/while/lstm_cell/mulMulDsequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:01sequential_5/lstm_5/while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђ▄
)sequential_5/lstm_5/while/lstm_cell/mul_1MulDsequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:01sequential_5/lstm_5/while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђ▄
)sequential_5/lstm_5/while/lstm_cell/mul_2MulDsequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:01sequential_5/lstm_5/while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђ▄
)sequential_5/lstm_5/while/lstm_cell/mul_3MulDsequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:01sequential_5/lstm_5/while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђu
3sequential_5/lstm_5/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Й
8sequential_5/lstm_5/while/lstm_cell/split/ReadVariableOpReadVariableOpCsequential_5_lstm_5_while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0ј
)sequential_5/lstm_5/while/lstm_cell/splitSplit<sequential_5/lstm_5/while/lstm_cell/split/split_dim:output:0@sequential_5/lstm_5/while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	ђ@:	ђ@:	ђ@:	ђ@*
	num_splitК
*sequential_5/lstm_5/while/lstm_cell/MatMulMatMul+sequential_5/lstm_5/while/lstm_cell/mul:z:02sequential_5/lstm_5/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:         @╦
,sequential_5/lstm_5/while/lstm_cell/MatMul_1MatMul-sequential_5/lstm_5/while/lstm_cell/mul_1:z:02sequential_5/lstm_5/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:         @╦
,sequential_5/lstm_5/while/lstm_cell/MatMul_2MatMul-sequential_5/lstm_5/while/lstm_cell/mul_2:z:02sequential_5/lstm_5/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:         @╦
,sequential_5/lstm_5/while/lstm_cell/MatMul_3MatMul-sequential_5/lstm_5/while/lstm_cell/mul_3:z:02sequential_5/lstm_5/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:         @w
5sequential_5/lstm_5/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : й
:sequential_5/lstm_5/while/lstm_cell/split_1/ReadVariableOpReadVariableOpEsequential_5_lstm_5_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ђ
+sequential_5/lstm_5/while/lstm_cell/split_1Split>sequential_5/lstm_5/while/lstm_cell/split_1/split_dim:output:0Bsequential_5/lstm_5/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitн
+sequential_5/lstm_5/while/lstm_cell/BiasAddBiasAdd4sequential_5/lstm_5/while/lstm_cell/MatMul:product:04sequential_5/lstm_5/while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:         @п
-sequential_5/lstm_5/while/lstm_cell/BiasAdd_1BiasAdd6sequential_5/lstm_5/while/lstm_cell/MatMul_1:product:04sequential_5/lstm_5/while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:         @п
-sequential_5/lstm_5/while/lstm_cell/BiasAdd_2BiasAdd6sequential_5/lstm_5/while/lstm_cell/MatMul_2:product:04sequential_5/lstm_5/while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:         @п
-sequential_5/lstm_5/while/lstm_cell/BiasAdd_3BiasAdd6sequential_5/lstm_5/while/lstm_cell/MatMul_3:product:04sequential_5/lstm_5/while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:         @└
)sequential_5/lstm_5/while/lstm_cell/mul_4Mul'sequential_5_lstm_5_while_placeholder_23sequential_5/lstm_5/while/lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @└
)sequential_5/lstm_5/while/lstm_cell/mul_5Mul'sequential_5_lstm_5_while_placeholder_23sequential_5/lstm_5/while/lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @└
)sequential_5/lstm_5/while/lstm_cell/mul_6Mul'sequential_5_lstm_5_while_placeholder_23sequential_5/lstm_5/while/lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @└
)sequential_5/lstm_5/while/lstm_cell/mul_7Mul'sequential_5_lstm_5_while_placeholder_23sequential_5/lstm_5/while/lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @▒
2sequential_5/lstm_5/while/lstm_cell/ReadVariableOpReadVariableOp=sequential_5_lstm_5_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@ђ*
dtype0ѕ
7sequential_5/lstm_5/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        і
9sequential_5/lstm_5/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   і
9sequential_5/lstm_5/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ъ
1sequential_5/lstm_5/while/lstm_cell/strided_sliceStridedSlice:sequential_5/lstm_5/while/lstm_cell/ReadVariableOp:value:0@sequential_5/lstm_5/while/lstm_cell/strided_slice/stack:output:0Bsequential_5/lstm_5/while/lstm_cell/strided_slice/stack_1:output:0Bsequential_5/lstm_5/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskМ
,sequential_5/lstm_5/while/lstm_cell/MatMul_4MatMul-sequential_5/lstm_5/while/lstm_cell/mul_4:z:0:sequential_5/lstm_5/while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:         @л
'sequential_5/lstm_5/while/lstm_cell/addAddV24sequential_5/lstm_5/while/lstm_cell/BiasAdd:output:06sequential_5/lstm_5/while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:         @Ћ
+sequential_5/lstm_5/while/lstm_cell/SigmoidSigmoid+sequential_5/lstm_5/while/lstm_cell/add:z:0*
T0*'
_output_shapes
:         @│
4sequential_5/lstm_5/while/lstm_cell/ReadVariableOp_1ReadVariableOp=sequential_5_lstm_5_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@ђ*
dtype0і
9sequential_5/lstm_5/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   ї
;sequential_5/lstm_5/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ђ   ї
;sequential_5/lstm_5/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Е
3sequential_5/lstm_5/while/lstm_cell/strided_slice_1StridedSlice<sequential_5/lstm_5/while/lstm_cell/ReadVariableOp_1:value:0Bsequential_5/lstm_5/while/lstm_cell/strided_slice_1/stack:output:0Dsequential_5/lstm_5/while/lstm_cell/strided_slice_1/stack_1:output:0Dsequential_5/lstm_5/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskН
,sequential_5/lstm_5/while/lstm_cell/MatMul_5MatMul-sequential_5/lstm_5/while/lstm_cell/mul_5:z:0<sequential_5/lstm_5/while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:         @н
)sequential_5/lstm_5/while/lstm_cell/add_1AddV26sequential_5/lstm_5/while/lstm_cell/BiasAdd_1:output:06sequential_5/lstm_5/while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:         @Ў
-sequential_5/lstm_5/while/lstm_cell/Sigmoid_1Sigmoid-sequential_5/lstm_5/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         @Й
)sequential_5/lstm_5/while/lstm_cell/mul_8Mul1sequential_5/lstm_5/while/lstm_cell/Sigmoid_1:y:0'sequential_5_lstm_5_while_placeholder_3*
T0*'
_output_shapes
:         @│
4sequential_5/lstm_5/while/lstm_cell/ReadVariableOp_2ReadVariableOp=sequential_5_lstm_5_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@ђ*
dtype0і
9sequential_5/lstm_5/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ђ   ї
;sequential_5/lstm_5/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   ї
;sequential_5/lstm_5/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Е
3sequential_5/lstm_5/while/lstm_cell/strided_slice_2StridedSlice<sequential_5/lstm_5/while/lstm_cell/ReadVariableOp_2:value:0Bsequential_5/lstm_5/while/lstm_cell/strided_slice_2/stack:output:0Dsequential_5/lstm_5/while/lstm_cell/strided_slice_2/stack_1:output:0Dsequential_5/lstm_5/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskН
,sequential_5/lstm_5/while/lstm_cell/MatMul_6MatMul-sequential_5/lstm_5/while/lstm_cell/mul_6:z:0<sequential_5/lstm_5/while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:         @н
)sequential_5/lstm_5/while/lstm_cell/add_2AddV26sequential_5/lstm_5/while/lstm_cell/BiasAdd_2:output:06sequential_5/lstm_5/while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:         @Љ
(sequential_5/lstm_5/while/lstm_cell/TanhTanh-sequential_5/lstm_5/while/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:         @┴
)sequential_5/lstm_5/while/lstm_cell/mul_9Mul/sequential_5/lstm_5/while/lstm_cell/Sigmoid:y:0,sequential_5/lstm_5/while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:         @┬
)sequential_5/lstm_5/while/lstm_cell/add_3AddV2-sequential_5/lstm_5/while/lstm_cell/mul_8:z:0-sequential_5/lstm_5/while/lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:         @│
4sequential_5/lstm_5/while/lstm_cell/ReadVariableOp_3ReadVariableOp=sequential_5_lstm_5_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@ђ*
dtype0і
9sequential_5/lstm_5/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   ї
;sequential_5/lstm_5/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ї
;sequential_5/lstm_5/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Е
3sequential_5/lstm_5/while/lstm_cell/strided_slice_3StridedSlice<sequential_5/lstm_5/while/lstm_cell/ReadVariableOp_3:value:0Bsequential_5/lstm_5/while/lstm_cell/strided_slice_3/stack:output:0Dsequential_5/lstm_5/while/lstm_cell/strided_slice_3/stack_1:output:0Dsequential_5/lstm_5/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskН
,sequential_5/lstm_5/while/lstm_cell/MatMul_7MatMul-sequential_5/lstm_5/while/lstm_cell/mul_7:z:0<sequential_5/lstm_5/while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:         @н
)sequential_5/lstm_5/while/lstm_cell/add_4AddV26sequential_5/lstm_5/while/lstm_cell/BiasAdd_3:output:06sequential_5/lstm_5/while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:         @Ў
-sequential_5/lstm_5/while/lstm_cell/Sigmoid_2Sigmoid-sequential_5/lstm_5/while/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:         @Њ
*sequential_5/lstm_5/while/lstm_cell/Tanh_1Tanh-sequential_5/lstm_5/while/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:         @к
*sequential_5/lstm_5/while/lstm_cell/mul_10Mul1sequential_5/lstm_5/while/lstm_cell/Sigmoid_2:y:0.sequential_5/lstm_5/while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:         @є
Dsequential_5/lstm_5/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ╗
>sequential_5/lstm_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_5_lstm_5_while_placeholder_1Msequential_5/lstm_5/while/TensorArrayV2Write/TensorListSetItem/index:output:0.sequential_5/lstm_5/while/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:жУмa
sequential_5/lstm_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :ў
sequential_5/lstm_5/while/addAddV2%sequential_5_lstm_5_while_placeholder(sequential_5/lstm_5/while/add/y:output:0*
T0*
_output_shapes
: c
!sequential_5/lstm_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :и
sequential_5/lstm_5/while/add_1AddV2@sequential_5_lstm_5_while_sequential_5_lstm_5_while_loop_counter*sequential_5/lstm_5/while/add_1/y:output:0*
T0*
_output_shapes
: Ћ
"sequential_5/lstm_5/while/IdentityIdentity#sequential_5/lstm_5/while/add_1:z:0^sequential_5/lstm_5/while/NoOp*
T0*
_output_shapes
: ║
$sequential_5/lstm_5/while/Identity_1IdentityFsequential_5_lstm_5_while_sequential_5_lstm_5_while_maximum_iterations^sequential_5/lstm_5/while/NoOp*
T0*
_output_shapes
: Ћ
$sequential_5/lstm_5/while/Identity_2Identity!sequential_5/lstm_5/while/add:z:0^sequential_5/lstm_5/while/NoOp*
T0*
_output_shapes
: ┬
$sequential_5/lstm_5/while/Identity_3IdentityNsequential_5/lstm_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_5/lstm_5/while/NoOp*
T0*
_output_shapes
: │
$sequential_5/lstm_5/while/Identity_4Identity.sequential_5/lstm_5/while/lstm_cell/mul_10:z:0^sequential_5/lstm_5/while/NoOp*
T0*'
_output_shapes
:         @▓
$sequential_5/lstm_5/while/Identity_5Identity-sequential_5/lstm_5/while/lstm_cell/add_3:z:0^sequential_5/lstm_5/while/NoOp*
T0*'
_output_shapes
:         @ј
sequential_5/lstm_5/while/NoOpNoOp3^sequential_5/lstm_5/while/lstm_cell/ReadVariableOp5^sequential_5/lstm_5/while/lstm_cell/ReadVariableOp_15^sequential_5/lstm_5/while/lstm_cell/ReadVariableOp_25^sequential_5/lstm_5/while/lstm_cell/ReadVariableOp_39^sequential_5/lstm_5/while/lstm_cell/split/ReadVariableOp;^sequential_5/lstm_5/while/lstm_cell/split_1/ReadVariableOp*
_output_shapes
 "Q
"sequential_5_lstm_5_while_identity+sequential_5/lstm_5/while/Identity:output:0"U
$sequential_5_lstm_5_while_identity_1-sequential_5/lstm_5/while/Identity_1:output:0"U
$sequential_5_lstm_5_while_identity_2-sequential_5/lstm_5/while/Identity_2:output:0"U
$sequential_5_lstm_5_while_identity_3-sequential_5/lstm_5/while/Identity_3:output:0"U
$sequential_5_lstm_5_while_identity_4-sequential_5/lstm_5/while/Identity_4:output:0"U
$sequential_5_lstm_5_while_identity_5-sequential_5/lstm_5/while/Identity_5:output:0"|
;sequential_5_lstm_5_while_lstm_cell_readvariableop_resource=sequential_5_lstm_5_while_lstm_cell_readvariableop_resource_0"ї
Csequential_5_lstm_5_while_lstm_cell_split_1_readvariableop_resourceEsequential_5_lstm_5_while_lstm_cell_split_1_readvariableop_resource_0"ѕ
Asequential_5_lstm_5_while_lstm_cell_split_readvariableop_resourceCsequential_5_lstm_5_while_lstm_cell_split_readvariableop_resource_0"ђ
=sequential_5_lstm_5_while_sequential_5_lstm_5_strided_slice_1?sequential_5_lstm_5_while_sequential_5_lstm_5_strided_slice_1_0"Э
ysequential_5_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_5_tensorarrayunstack_tensorlistfromtensor{sequential_5_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_5_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2h
2sequential_5/lstm_5/while/lstm_cell/ReadVariableOp2sequential_5/lstm_5/while/lstm_cell/ReadVariableOp2l
4sequential_5/lstm_5/while/lstm_cell/ReadVariableOp_14sequential_5/lstm_5/while/lstm_cell/ReadVariableOp_12l
4sequential_5/lstm_5/while/lstm_cell/ReadVariableOp_24sequential_5/lstm_5/while/lstm_cell/ReadVariableOp_22l
4sequential_5/lstm_5/while/lstm_cell/ReadVariableOp_34sequential_5/lstm_5/while/lstm_cell/ReadVariableOp_32t
8sequential_5/lstm_5/while/lstm_cell/split/ReadVariableOp8sequential_5/lstm_5/while/lstm_cell/split/ReadVariableOp2x
:sequential_5/lstm_5/while/lstm_cell/split_1/ReadVariableOp:sequential_5/lstm_5/while/lstm_cell/split_1/ReadVariableOp:^ Z

_output_shapes
: 
@
_user_specified_name(&sequential_5/lstm_5/while/loop_counter:d`

_output_shapes
: 
F
_user_specified_name.,sequential_5/lstm_5/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:[W

_output_shapes
: 
=
_user_specified_name%#sequential_5/lstm_5/strided_slice_1:so

_output_shapes
: 
U
_user_specified_name=;sequential_5/lstm_5/TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
А┬
╦
A__inference_lstm_5_layer_call_and_return_conditional_losses_63391

inputs;
'lstm_cell_split_readvariableop_resource:
ђђ8
)lstm_cell_split_1_readvariableop_resource:	ђ4
!lstm_cell_readvariableop_resource:	@ђ
identityѕбlstm_cell/ReadVariableOpбlstm_cell/ReadVariableOp_1бlstm_cell/ReadVariableOp_2бlstm_cell/ReadVariableOp_3бlstm_cell/split/ReadVariableOpб lstm_cell/split_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          o
	transpose	Transposeinputstranspose/perm:output:0*
T0*-
_output_shapes
:У         ђR
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ђ*
shrink_axis_maskl
lstm_cell/ones_likeOnesLikestrided_slice_2:output:0*
T0*(
_output_shapes
:         ђ\
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?і
lstm_cell/dropout/MulMullstm_cell/ones_like:y:0 lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:         ђl
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:y:0*
T0*
_output_shapes
::ь¤А
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0e
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>┼
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ^
lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ╝
lstm_cell/dropout/SelectV2SelectV2"lstm_cell/dropout/GreaterEqual:z:0lstm_cell/dropout/Mul:z:0"lstm_cell/dropout/Const_1:output:0*
T0*(
_output_shapes
:         ђ^
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?ј
lstm_cell/dropout_1/MulMullstm_cell/ones_like:y:0"lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:         ђn
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:y:0*
T0*
_output_shapes
::ь¤Ц
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0g
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>╦
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ`
lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ─
lstm_cell/dropout_1/SelectV2SelectV2$lstm_cell/dropout_1/GreaterEqual:z:0lstm_cell/dropout_1/Mul:z:0$lstm_cell/dropout_1/Const_1:output:0*
T0*(
_output_shapes
:         ђ^
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?ј
lstm_cell/dropout_2/MulMullstm_cell/ones_like:y:0"lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:         ђn
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:y:0*
T0*
_output_shapes
::ь¤Ц
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0g
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>╦
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ`
lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ─
lstm_cell/dropout_2/SelectV2SelectV2$lstm_cell/dropout_2/GreaterEqual:z:0lstm_cell/dropout_2/Mul:z:0$lstm_cell/dropout_2/Const_1:output:0*
T0*(
_output_shapes
:         ђ^
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?ј
lstm_cell/dropout_3/MulMullstm_cell/ones_like:y:0"lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:         ђn
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:y:0*
T0*
_output_shapes
::ь¤Ц
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0g
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>╦
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ`
lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ─
lstm_cell/dropout_3/SelectV2SelectV2$lstm_cell/dropout_3/GreaterEqual:z:0lstm_cell/dropout_3/Mul:z:0$lstm_cell/dropout_3/Const_1:output:0*
T0*(
_output_shapes
:         ђc
lstm_cell/ones_like_1OnesLikezeros:output:0*
T0*'
_output_shapes
:         @^
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?Ј
lstm_cell/dropout_4/MulMullstm_cell/ones_like_1:y:0"lstm_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:         @p
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::ь¤ц
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0g
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>╩
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @`
lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ├
lstm_cell/dropout_4/SelectV2SelectV2$lstm_cell/dropout_4/GreaterEqual:z:0lstm_cell/dropout_4/Mul:z:0$lstm_cell/dropout_4/Const_1:output:0*
T0*'
_output_shapes
:         @^
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?Ј
lstm_cell/dropout_5/MulMullstm_cell/ones_like_1:y:0"lstm_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:         @p
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::ь¤ц
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0g
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>╩
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @`
lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ├
lstm_cell/dropout_5/SelectV2SelectV2$lstm_cell/dropout_5/GreaterEqual:z:0lstm_cell/dropout_5/Mul:z:0$lstm_cell/dropout_5/Const_1:output:0*
T0*'
_output_shapes
:         @^
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?Ј
lstm_cell/dropout_6/MulMullstm_cell/ones_like_1:y:0"lstm_cell/dropout_6/Const:output:0*
T0*'
_output_shapes
:         @p
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::ь¤ц
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0g
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>╩
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @`
lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ├
lstm_cell/dropout_6/SelectV2SelectV2$lstm_cell/dropout_6/GreaterEqual:z:0lstm_cell/dropout_6/Mul:z:0$lstm_cell/dropout_6/Const_1:output:0*
T0*'
_output_shapes
:         @^
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?Ј
lstm_cell/dropout_7/MulMullstm_cell/ones_like_1:y:0"lstm_cell/dropout_7/Const:output:0*
T0*'
_output_shapes
:         @p
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::ь¤ц
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0g
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>╩
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @`
lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ├
lstm_cell/dropout_7/SelectV2SelectV2$lstm_cell/dropout_7/GreaterEqual:z:0lstm_cell/dropout_7/Mul:z:0$lstm_cell/dropout_7/Const_1:output:0*
T0*'
_output_shapes
:         @є
lstm_cell/mulMulstrided_slice_2:output:0#lstm_cell/dropout/SelectV2:output:0*
T0*(
_output_shapes
:         ђі
lstm_cell/mul_1Mulstrided_slice_2:output:0%lstm_cell/dropout_1/SelectV2:output:0*
T0*(
_output_shapes
:         ђі
lstm_cell/mul_2Mulstrided_slice_2:output:0%lstm_cell/dropout_2/SelectV2:output:0*
T0*(
_output_shapes
:         ђі
lstm_cell/mul_3Mulstrided_slice_2:output:0%lstm_cell/dropout_3/SelectV2:output:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ѕ
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0└
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	ђ@:	ђ@:	ђ@:	ђ@*
	num_splity
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*'
_output_shapes
:         @}
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*'
_output_shapes
:         @}
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*'
_output_shapes
:         @}
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*'
_output_shapes
:         @]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Є
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype0▓
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitє
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:         @і
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:         @і
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:         @і
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:         @
lstm_cell/mul_4Mulzeros:output:0%lstm_cell/dropout_4/SelectV2:output:0*
T0*'
_output_shapes
:         @
lstm_cell/mul_5Mulzeros:output:0%lstm_cell/dropout_5/SelectV2:output:0*
T0*'
_output_shapes
:         @
lstm_cell/mul_6Mulzeros:output:0%lstm_cell/dropout_6/SelectV2:output:0*
T0*'
_output_shapes
:         @
lstm_cell/mul_7Mulzeros:output:0%lstm_cell/dropout_7/SelectV2:output:0*
T0*'
_output_shapes
:         @{
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ю
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЁ
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:         @ѓ
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:         @a
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:         @}
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ђ   r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Д
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЄ
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:         @є
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:         @e
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:         @s
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         @}
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ђ   r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Д
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЄ
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:         @є
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:         @]
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*'
_output_shapes
:         @s
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:         @t
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:         @}
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Д
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЄ
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:         @є
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:         @e
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:         @_
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*'
_output_shapes
:         @x
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : л
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_63196*
condR
while_cond_63195*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         @Я
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         Уђ: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:         Уђ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
й	
Й
while_cond_64635
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_64635___redundant_placeholder03
/while_while_cond_64635___redundant_placeholder13
/while_while_cond_64635___redundant_placeholder23
/while_while_cond_64635___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
─
├
G__inference_sequential_5_layer_call_and_return_conditional_losses_63434
embedding_5_input%
embedding_5_63024:
ѕ'ђ 
lstm_5_63392:
ђђ
lstm_5_63394:	ђ
lstm_5_63396:	@ђ
dense_5_63409:@
dense_5_63411:
identityѕбdense_5/StatefulPartitionedCallб!dropout_5/StatefulPartitionedCallб#embedding_5/StatefulPartitionedCallбlstm_5/StatefulPartitionedCallш
#embedding_5/StatefulPartitionedCallStatefulPartitionedCallembedding_5_inputembedding_5_63024*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         Уђ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_embedding_5_layer_call_and_return_conditional_losses_63023Џ
lstm_5/StatefulPartitionedCallStatefulPartitionedCall,embedding_5/StatefulPartitionedCall:output:0lstm_5_63392lstm_5_63394lstm_5_63396*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_63391і
dense_5/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0dense_5_63409dense_5_63411*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_63408в
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_63425с
activation_5/PartitionedCallPartitionedCall*dropout_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_63431t
IdentityIdentity%activation_5/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         »
NoOpNoOp ^dense_5/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall$^embedding_5/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         У: : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2J
#embedding_5/StatefulPartitionedCall#embedding_5/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall:[ W
(
_output_shapes
:         У
+
_user_specified_nameembedding_5_input:%!

_user_specified_name63024:%!

_user_specified_name63392:%!

_user_specified_name63394:%!

_user_specified_name63396:%!

_user_specified_name63409:%!

_user_specified_name63411
Ї{
╦
A__inference_lstm_5_layer_call_and_return_conditional_losses_65068

inputs;
'lstm_cell_split_readvariableop_resource:
ђђ8
)lstm_cell_split_1_readvariableop_resource:	ђ4
!lstm_cell_readvariableop_resource:	@ђ
identityѕбlstm_cell/ReadVariableOpбlstm_cell/ReadVariableOp_1бlstm_cell/ReadVariableOp_2бlstm_cell/ReadVariableOp_3бlstm_cell/split/ReadVariableOpб lstm_cell/split_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          o
	transpose	Transposeinputstranspose/perm:output:0*
T0*-
_output_shapes
:У         ђR
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ђ*
shrink_axis_maskl
lstm_cell/ones_likeOnesLikestrided_slice_2:output:0*
T0*(
_output_shapes
:         ђc
lstm_cell/ones_like_1OnesLikezeros:output:0*
T0*'
_output_shapes
:         @z
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђ|
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђ|
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђ|
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ѕ
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0└
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	ђ@:	ђ@:	ђ@:	ђ@*
	num_splity
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*'
_output_shapes
:         @}
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*'
_output_shapes
:         @}
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*'
_output_shapes
:         @}
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*'
_output_shapes
:         @]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Є
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype0▓
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitє
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:         @і
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:         @і
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:         @і
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:         @s
lstm_cell/mul_4Mulzeros:output:0lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @s
lstm_cell/mul_5Mulzeros:output:0lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @s
lstm_cell/mul_6Mulzeros:output:0lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @s
lstm_cell/mul_7Mulzeros:output:0lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @{
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ю
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЁ
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:         @ѓ
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:         @a
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:         @}
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ђ   r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Д
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЄ
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:         @є
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:         @e
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:         @s
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         @}
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ђ   r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Д
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЄ
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:         @є
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:         @]
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*'
_output_shapes
:         @s
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:         @t
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:         @}
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Д
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЄ
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:         @є
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:         @e
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:         @_
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*'
_output_shapes
:         @x
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : л
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_64937*
condR
while_cond_64936*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         @Я
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         Уђ: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:         Уђ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ёs
├
!__inference__traced_restore_65625
file_prefix;
'assignvariableop_embedding_5_embeddings:
ѕ'ђ3
!assignvariableop_1_dense_5_kernel:@-
assignvariableop_2_dense_5_bias:>
*assignvariableop_3_lstm_5_lstm_cell_kernel:
ђђG
4assignvariableop_4_lstm_5_lstm_cell_recurrent_kernel:	@ђ7
(assignvariableop_5_lstm_5_lstm_cell_bias:	ђ&
assignvariableop_6_iteration:	 *
 assignvariableop_7_learning_rate: D
0assignvariableop_8_adam_m_embedding_5_embeddings:
ѕ'ђD
0assignvariableop_9_adam_v_embedding_5_embeddings:
ѕ'ђF
2assignvariableop_10_adam_m_lstm_5_lstm_cell_kernel:
ђђF
2assignvariableop_11_adam_v_lstm_5_lstm_cell_kernel:
ђђO
<assignvariableop_12_adam_m_lstm_5_lstm_cell_recurrent_kernel:	@ђO
<assignvariableop_13_adam_v_lstm_5_lstm_cell_recurrent_kernel:	@ђ?
0assignvariableop_14_adam_m_lstm_5_lstm_cell_bias:	ђ?
0assignvariableop_15_adam_v_lstm_5_lstm_cell_bias:	ђ;
)assignvariableop_16_adam_m_dense_5_kernel:@;
)assignvariableop_17_adam_v_dense_5_kernel:@5
'assignvariableop_18_adam_m_dense_5_bias:5
'assignvariableop_19_adam_v_dense_5_bias:%
assignvariableop_20_total_1: %
assignvariableop_21_count_1: #
assignvariableop_22_total: #
assignvariableop_23_count: 
identity_25ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9Н

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ч	
valueы	BЬ	B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHб
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B Џ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOpAssignVariableOp'assignvariableop_embedding_5_embeddingsIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_5_kernelIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_5_biasIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_3AssignVariableOp*assignvariableop_3_lstm_5_lstm_cell_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:╦
AssignVariableOp_4AssignVariableOp4assignvariableop_4_lstm_5_lstm_cell_recurrent_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_5AssignVariableOp(assignvariableop_5_lstm_5_lstm_cell_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:│
AssignVariableOp_6AssignVariableOpassignvariableop_6_iterationIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_7AssignVariableOp assignvariableop_7_learning_rateIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_8AssignVariableOp0assignvariableop_8_adam_m_embedding_5_embeddingsIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_9AssignVariableOp0assignvariableop_9_adam_v_embedding_5_embeddingsIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:╦
AssignVariableOp_10AssignVariableOp2assignvariableop_10_adam_m_lstm_5_lstm_cell_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:╦
AssignVariableOp_11AssignVariableOp2assignvariableop_11_adam_v_lstm_5_lstm_cell_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_12AssignVariableOp<assignvariableop_12_adam_m_lstm_5_lstm_cell_recurrent_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_13AssignVariableOp<assignvariableop_13_adam_v_lstm_5_lstm_cell_recurrent_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:╔
AssignVariableOp_14AssignVariableOp0assignvariableop_14_adam_m_lstm_5_lstm_cell_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:╔
AssignVariableOp_15AssignVariableOp0assignvariableop_15_adam_v_lstm_5_lstm_cell_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_m_dense_5_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_v_dense_5_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_m_dense_5_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_v_dense_5_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_20AssignVariableOpassignvariableop_20_total_1Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_1Identity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ▀
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: е
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_25Identity_25:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2: : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:62
0
_user_specified_nameembedding_5/embeddings:.*
(
_user_specified_namedense_5/kernel:,(
&
_user_specified_namedense_5/bias:73
1
_user_specified_namelstm_5/lstm_cell/kernel:A=
;
_user_specified_name#!lstm_5/lstm_cell/recurrent_kernel:51
/
_user_specified_namelstm_5/lstm_cell/bias:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namelearning_rate:=	9
7
_user_specified_nameAdam/m/embedding_5/embeddings:=
9
7
_user_specified_nameAdam/v/embedding_5/embeddings:>:
8
_user_specified_name Adam/m/lstm_5/lstm_cell/kernel:>:
8
_user_specified_name Adam/v/lstm_5/lstm_cell/kernel:HD
B
_user_specified_name*(Adam/m/lstm_5/lstm_cell/recurrent_kernel:HD
B
_user_specified_name*(Adam/v/lstm_5/lstm_cell/recurrent_kernel:<8
6
_user_specified_nameAdam/m/lstm_5/lstm_cell/bias:<8
6
_user_specified_nameAdam/v/lstm_5/lstm_cell/bias:51
/
_user_specified_nameAdam/m/dense_5/kernel:51
/
_user_specified_nameAdam/v/dense_5/kernel:3/
-
_user_specified_nameAdam/m/dense_5/bias:3/
-
_user_specified_nameAdam/v/dense_5/bias:'#
!
_user_specified_name	total_1:'#
!
_user_specified_name	count_1:%!

_user_specified_nametotal:%!

_user_specified_namecount
Г{
═
A__inference_lstm_5_layer_call_and_return_conditional_losses_64466
inputs_0;
'lstm_cell_split_readvariableop_resource:
ђђ8
)lstm_cell_split_1_readvariableop_resource:	ђ4
!lstm_cell_readvariableop_resource:	@ђ
identityѕбlstm_cell/ReadVariableOpбlstm_cell/ReadVariableOp_1бlstm_cell/ReadVariableOp_2бlstm_cell/ReadVariableOp_3бlstm_cell/split/ReadVariableOpб lstm_cell/split_1/ReadVariableOpбwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:                  ђR
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ђ*
shrink_axis_maskl
lstm_cell/ones_likeOnesLikestrided_slice_2:output:0*
T0*(
_output_shapes
:         ђc
lstm_cell/ones_like_1OnesLikezeros:output:0*
T0*'
_output_shapes
:         @z
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђ|
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђ|
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђ|
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ѕ
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0└
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	ђ@:	ђ@:	ђ@:	ђ@*
	num_splity
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*'
_output_shapes
:         @}
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*'
_output_shapes
:         @}
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*'
_output_shapes
:         @}
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*'
_output_shapes
:         @]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Є
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype0▓
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitє
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:         @і
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:         @і
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:         @і
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:         @s
lstm_cell/mul_4Mulzeros:output:0lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @s
lstm_cell/mul_5Mulzeros:output:0lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @s
lstm_cell/mul_6Mulzeros:output:0lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @s
lstm_cell/mul_7Mulzeros:output:0lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @{
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ю
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЁ
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:         @ѓ
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:         @a
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:         @}
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ђ   r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Д
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЄ
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:         @є
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:         @e
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:         @s
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         @}
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ђ   r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Д
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЄ
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:         @є
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:         @]
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*'
_output_shapes
:         @s
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:         @t
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:         @}
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Д
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЄ
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:         @є
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:         @e
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:         @_
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*'
_output_shapes
:         @x
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : л
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_64335*
condR
while_cond_64334*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         @Я
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ђ: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:                  ђ
"
_user_specified_name
inputs_0:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
¤p
Ђ	
while_body_64937
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
/while_lstm_cell_split_readvariableop_resource_0:
ђђ@
1while_lstm_cell_split_1_readvariableop_resource_0:	ђ<
)while_lstm_cell_readvariableop_resource_0:	@ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
-while_lstm_cell_split_readvariableop_resource:
ђђ>
/while_lstm_cell_split_1_readvariableop_resource:	ђ:
'while_lstm_cell_readvariableop_resource:	@ђѕбwhile/lstm_cell/ReadVariableOpб while/lstm_cell/ReadVariableOp_1б while/lstm_cell/ReadVariableOp_2б while/lstm_cell/ReadVariableOp_3б$while/lstm_cell/split/ReadVariableOpб&while/lstm_cell/split_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   Д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ђ*
element_dtype0і
while/lstm_cell/ones_likeOnesLike0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*(
_output_shapes
:         ђn
while/lstm_cell/ones_like_1OnesLikewhile_placeholder_2*
T0*'
_output_shapes
:         @ъ
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђа
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђа
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђа
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ќ
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0м
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	ђ@:	ђ@:	ђ@:	ђ@*
	num_splitІ
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:         @Ј
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:         @Ј
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:         @Ј
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:         @c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ћ
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0─
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitў
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:         @ю
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:         @ю
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:         @ю
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:         @ё
while/lstm_cell/mul_4Mulwhile_placeholder_2while/lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @ё
while/lstm_cell/mul_5Mulwhile_placeholder_2while/lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @ё
while/lstm_cell/mul_6Mulwhile_placeholder_2while/lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @ё
while/lstm_cell/mul_7Mulwhile_placeholder_2while/lstm_cell/ones_like_1:y:0*
T0*'
_output_shapes
:         @Ѕ
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@ђ*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╗
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЌ
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:         @ћ
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:         @m
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:         @І
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@ђ*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ђ   x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┼
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЎ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:         @ў
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:         @q
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         @ѓ
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @І
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@ђ*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ђ   x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┼
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЎ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:         @ў
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:         @i
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:         @Ё
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:         @є
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:         @І
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@ђ*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┼
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЎ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:         @ў
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:         @q
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:         @k
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:         @і
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:         @r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:         @v
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:         @ѓ

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
╩
c
G__inference_activation_5_layer_call_and_return_conditional_losses_63431

inputs
identityL
SigmoidSigmoidinputs*
T0*'
_output_shapes
:         S
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╔
┤
&__inference_lstm_5_layer_call_fn_63864

inputs
unknown:
ђђ
	unknown_0:	ђ
	unknown_1:	@ђ
identityѕбStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_63676o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         Уђ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:         Уђ
 
_user_specified_nameinputs:%!

_user_specified_name63856:%!

_user_specified_name63858:%!

_user_specified_name63860
и
з
)__inference_lstm_cell_layer_call_fn_65158

inputs
states_0
states_1
unknown:
ђђ
	unknown_0:	ђ
	unknown_1:	@ђ
identity

identity_1

identity_2ѕбStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         @:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_62871o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         @q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         ђ:         @:         @: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs:QM
'
_output_shapes
:         @
"
_user_specified_name
states_0:QM
'
_output_shapes
:         @
"
_user_specified_name
states_1:%!

_user_specified_name65146:%!

_user_specified_name65148:%!

_user_specified_name65150
й	
Й
while_cond_62885
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_62885___redundant_placeholder03
/while_while_cond_62885___redundant_placeholder13
/while_while_cond_62885___redundant_placeholder23
/while_while_cond_62885___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:"ьL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*─
serving_default░
P
embedding_5_input;
#serving_default_embedding_5_input:0         У@
activation_50
StatefulPartitionedCall:0         tensorflow/serving/predict:Ѓ┼
ш
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
х
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
┌
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec"
_tf_keras_rnn_layer
╗
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias"
_tf_keras_layer
╝
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses
-_random_generator"
_tf_keras_layer
Ц
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses"
_tf_keras_layer
J
0
41
52
63
%4
&5"
trackable_list_wrapper
J
0
41
52
63
%4
&5"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
╦
<trace_0
=trace_12ћ
,__inference_sequential_5_layer_call_fn_63714
,__inference_sequential_5_layer_call_fn_63731х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z<trace_0z=trace_1
Ђ
>trace_0
?trace_12╩
G__inference_sequential_5_layer_call_and_return_conditional_losses_63434
G__inference_sequential_5_layer_call_and_return_conditional_losses_63697х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z>trace_0z?trace_1
НBм
 __inference__wrapped_model_62506embedding_5_input"ў
Љ▓Ї
FullArgSpec
argsџ

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ю
@
_variables
A_iterations
B_learning_rate
C_index_dict
D
_momentums
E_velocities
F_update_step_xla"
experimentalOptimizer
,
Gserving_default"
signature_map
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
т
Mtrace_02╚
+__inference_embedding_5_layer_call_fn_63811ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zMtrace_0
ђ
Ntrace_02с
F__inference_embedding_5_layer_call_and_return_conditional_losses_63820ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zNtrace_0
*:(
ѕ'ђ2embedding_5/embeddings
5
40
51
62"
trackable_list_wrapper
5
40
51
62"
trackable_list_wrapper
 "
trackable_list_wrapper
╣

Ostates
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
п
Utrace_0
Vtrace_1
Wtrace_2
Xtrace_32ь
&__inference_lstm_5_layer_call_fn_63831
&__inference_lstm_5_layer_call_fn_63842
&__inference_lstm_5_layer_call_fn_63853
&__inference_lstm_5_layer_call_fn_63864╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zUtrace_0zVtrace_1zWtrace_2zXtrace_3
─
Ytrace_0
Ztrace_1
[trace_2
\trace_32┘
A__inference_lstm_5_layer_call_and_return_conditional_losses_64229
A__inference_lstm_5_layer_call_and_return_conditional_losses_64466
A__inference_lstm_5_layer_call_and_return_conditional_losses_64831
A__inference_lstm_5_layer_call_and_return_conditional_losses_65068╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zYtrace_0zZtrace_1z[trace_2z\trace_3
"
_generic_user_object
Э
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses
c_random_generator
d
state_size

4kernel
5recurrent_kernel
6bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
р
jtrace_02─
'__inference_dense_5_layer_call_fn_65077ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zjtrace_0
Ч
ktrace_02▀
B__inference_dense_5_layer_call_and_return_conditional_losses_65087ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zktrace_0
 :@2dense_5/kernel
:2dense_5/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Г
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
╣
qtrace_0
rtrace_12ѓ
)__inference_dropout_5_layer_call_fn_65092
)__inference_dropout_5_layer_call_fn_65097Е
б▓ъ
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zqtrace_0zrtrace_1
№
strace_0
ttrace_12И
D__inference_dropout_5_layer_call_and_return_conditional_losses_65109
D__inference_dropout_5_layer_call_and_return_conditional_losses_65114Е
б▓ъ
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zstrace_0zttrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Г
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
Т
ztrace_02╔
,__inference_activation_5_layer_call_fn_65119ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zztrace_0
Ђ
{trace_02С
G__inference_activation_5_layer_call_and_return_conditional_losses_65124ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z{trace_0
+:)
ђђ2lstm_5/lstm_cell/kernel
4:2	@ђ2!lstm_5/lstm_cell/recurrent_kernel
$:"ђ2lstm_5/lstm_cell/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
шBЫ
,__inference_sequential_5_layer_call_fn_63714embedding_5_input"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
шBЫ
,__inference_sequential_5_layer_call_fn_63731embedding_5_input"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
љBЇ
G__inference_sequential_5_layer_call_and_return_conditional_losses_63434embedding_5_input"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
љBЇ
G__inference_sequential_5_layer_call_and_return_conditional_losses_63697embedding_5_input"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ѕ
A0
~1
2
ђ3
Ђ4
ѓ5
Ѓ6
ё7
Ё8
є9
Є10
ѕ11
Ѕ12"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
O
~0
ђ1
ѓ2
ё3
є4
ѕ5"
trackable_list_wrapper
O
0
Ђ1
Ѓ2
Ё3
Є4
Ѕ5"
trackable_list_wrapper
х2▓»
д▓б
FullArgSpec*
args"џ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
сBЯ
#__inference_signature_wrapper_63804embedding_5_input"Б
ю▓ў
FullArgSpec
argsџ 
varargs
 
varkw
 
defaults
 &

kwonlyargsџ
jembedding_5_input
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
НBм
+__inference_embedding_5_layer_call_fn_63811inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­Bь
F__inference_embedding_5_layer_call_and_return_conditional_losses_63820inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
эBЗ
&__inference_lstm_5_layer_call_fn_63831inputs_0"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
эBЗ
&__inference_lstm_5_layer_call_fn_63842inputs_0"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
шBЫ
&__inference_lstm_5_layer_call_fn_63853inputs"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
шBЫ
&__inference_lstm_5_layer_call_fn_63864inputs"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
њBЈ
A__inference_lstm_5_layer_call_and_return_conditional_losses_64229inputs_0"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
њBЈ
A__inference_lstm_5_layer_call_and_return_conditional_losses_64466inputs_0"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
љBЇ
A__inference_lstm_5_layer_call_and_return_conditional_losses_64831inputs"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
љBЇ
A__inference_lstm_5_layer_call_and_return_conditional_losses_65068inputs"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
5
40
51
62"
trackable_list_wrapper
5
40
51
62"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
іnon_trainable_variables
Іlayers
їmetrics
 Їlayer_regularization_losses
јlayer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
К
Јtrace_0
љtrace_12ї
)__inference_lstm_cell_layer_call_fn_65141
)__inference_lstm_cell_layer_call_fn_65158│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЈtrace_0zљtrace_1
§
Љtrace_0
њtrace_12┬
D__inference_lstm_cell_layer_call_and_return_conditional_losses_65300
D__inference_lstm_cell_layer_call_and_return_conditional_losses_65378│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЉtrace_0zњtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЛB╬
'__inference_dense_5_layer_call_fn_65077inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ВBж
B__inference_dense_5_layer_call_and_return_conditional_losses_65087inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▀B▄
)__inference_dropout_5_layer_call_fn_65092inputs"ц
Ю▓Ў
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
▀B▄
)__inference_dropout_5_layer_call_fn_65097inputs"ц
Ю▓Ў
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
D__inference_dropout_5_layer_call_and_return_conditional_losses_65109inputs"ц
Ю▓Ў
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
D__inference_dropout_5_layer_call_and_return_conditional_losses_65114inputs"ц
Ю▓Ў
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
оBМ
,__inference_activation_5_layer_call_fn_65119inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ыBЬ
G__inference_activation_5_layer_call_and_return_conditional_losses_65124inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
R
Њ	variables
ћ	keras_api

Ћtotal

ќcount"
_tf_keras_metric
c
Ќ	variables
ў	keras_api

Ўtotal

џcount
Џ
_fn_kwargs"
_tf_keras_metric
/:-
ѕ'ђ2Adam/m/embedding_5/embeddings
/:-
ѕ'ђ2Adam/v/embedding_5/embeddings
0:.
ђђ2Adam/m/lstm_5/lstm_cell/kernel
0:.
ђђ2Adam/v/lstm_5/lstm_cell/kernel
9:7	@ђ2(Adam/m/lstm_5/lstm_cell/recurrent_kernel
9:7	@ђ2(Adam/v/lstm_5/lstm_cell/recurrent_kernel
):'ђ2Adam/m/lstm_5/lstm_cell/bias
):'ђ2Adam/v/lstm_5/lstm_cell/bias
%:#@2Adam/m/dense_5/kernel
%:#@2Adam/v/dense_5/kernel
:2Adam/m/dense_5/bias
:2Adam/v/dense_5/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
§BЩ
)__inference_lstm_cell_layer_call_fn_65141inputsstates_0states_1"«
Д▓Б
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
§BЩ
)__inference_lstm_cell_layer_call_fn_65158inputsstates_0states_1"«
Д▓Б
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ўBЋ
D__inference_lstm_cell_layer_call_and_return_conditional_losses_65300inputsstates_0states_1"«
Д▓Б
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ўBЋ
D__inference_lstm_cell_layer_call_and_return_conditional_losses_65378inputsstates_0states_1"«
Д▓Б
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
0
Ћ0
ќ1"
trackable_list_wrapper
.
Њ	variables"
_generic_user_object
:  (2total
:  (2count
0
Ў0
џ1"
trackable_list_wrapper
.
Ќ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperД
 __inference__wrapped_model_62506ѓ465%&;б8
1б.
,і)
embedding_5_input         У
ф ";ф8
6
activation_5&і#
activation_5         ф
G__inference_activation_5_layer_call_and_return_conditional_losses_65124_/б,
%б"
 і
inputs         
ф ",б)
"і
tensor_0         
џ ё
,__inference_activation_5_layer_call_fn_65119T/б,
%б"
 і
inputs         
ф "!і
unknown         Е
B__inference_dense_5_layer_call_and_return_conditional_losses_65087c%&/б,
%б"
 і
inputs         @
ф ",б)
"і
tensor_0         
џ Ѓ
'__inference_dense_5_layer_call_fn_65077X%&/б,
%б"
 і
inputs         @
ф "!і
unknown         Ф
D__inference_dropout_5_layer_call_and_return_conditional_losses_65109c3б0
)б&
 і
inputs         
p
ф ",б)
"і
tensor_0         
џ Ф
D__inference_dropout_5_layer_call_and_return_conditional_losses_65114c3б0
)б&
 і
inputs         
p 
ф ",б)
"і
tensor_0         
џ Ё
)__inference_dropout_5_layer_call_fn_65092X3б0
)б&
 і
inputs         
p
ф "!і
unknown         Ё
)__inference_dropout_5_layer_call_fn_65097X3б0
)б&
 і
inputs         
p 
ф "!і
unknown         │
F__inference_embedding_5_layer_call_and_return_conditional_losses_63820i0б-
&б#
!і
inputs         У
ф "2б/
(і%
tensor_0         Уђ
џ Ї
+__inference_embedding_5_layer_call_fn_63811^0б-
&б#
!і
inputs         У
ф "'і$
unknown         Уђ╦
A__inference_lstm_5_layer_call_and_return_conditional_losses_64229Ё465PбM
FбC
5џ2
0і-
inputs_0                  ђ

 
p

 
ф ",б)
"і
tensor_0         @
џ ╦
A__inference_lstm_5_layer_call_and_return_conditional_losses_64466Ё465PбM
FбC
5џ2
0і-
inputs_0                  ђ

 
p 

 
ф ",б)
"і
tensor_0         @
џ ╗
A__inference_lstm_5_layer_call_and_return_conditional_losses_64831v465Aб>
7б4
&і#
inputs         Уђ

 
p

 
ф ",б)
"і
tensor_0         @
џ ╗
A__inference_lstm_5_layer_call_and_return_conditional_losses_65068v465Aб>
7б4
&і#
inputs         Уђ

 
p 

 
ф ",б)
"і
tensor_0         @
џ ц
&__inference_lstm_5_layer_call_fn_63831z465PбM
FбC
5џ2
0і-
inputs_0                  ђ

 
p

 
ф "!і
unknown         @ц
&__inference_lstm_5_layer_call_fn_63842z465PбM
FбC
5џ2
0і-
inputs_0                  ђ

 
p 

 
ф "!і
unknown         @Ћ
&__inference_lstm_5_layer_call_fn_63853k465Aб>
7б4
&і#
inputs         Уђ

 
p

 
ф "!і
unknown         @Ћ
&__inference_lstm_5_layer_call_fn_63864k465Aб>
7б4
&і#
inputs         Уђ

 
p 

 
ф "!і
unknown         @я
D__inference_lstm_cell_layer_call_and_return_conditional_losses_65300Ћ465Ђб~
wбt
!і
inputs         ђ
KбH
"і
states_0         @
"і
states_1         @
p
ф "ЅбЁ
~б{
$і!

tensor_0_0         @
SџP
&і#
tensor_0_1_0         @
&і#
tensor_0_1_1         @
џ я
D__inference_lstm_cell_layer_call_and_return_conditional_losses_65378Ћ465Ђб~
wбt
!і
inputs         ђ
KбH
"і
states_0         @
"і
states_1         @
p 
ф "ЅбЁ
~б{
$і!

tensor_0_0         @
SџP
&і#
tensor_0_1_0         @
&і#
tensor_0_1_1         @
џ ▒
)__inference_lstm_cell_layer_call_fn_65141Ѓ465Ђб~
wбt
!і
inputs         ђ
KбH
"і
states_0         @
"і
states_1         @
p
ф "xбu
"і
tensor_0         @
OџL
$і!

tensor_1_0         @
$і!

tensor_1_1         @▒
)__inference_lstm_cell_layer_call_fn_65158Ѓ465Ђб~
wбt
!і
inputs         ђ
KбH
"і
states_0         @
"і
states_1         @
p 
ф "xбu
"і
tensor_0         @
OџL
$і!

tensor_1_0         @
$і!

tensor_1_1         @к
G__inference_sequential_5_layer_call_and_return_conditional_losses_63434{465%&Cб@
9б6
,і)
embedding_5_input         У
p

 
ф ",б)
"і
tensor_0         
џ к
G__inference_sequential_5_layer_call_and_return_conditional_losses_63697{465%&Cб@
9б6
,і)
embedding_5_input         У
p 

 
ф ",б)
"і
tensor_0         
џ а
,__inference_sequential_5_layer_call_fn_63714p465%&Cб@
9б6
,і)
embedding_5_input         У
p

 
ф "!і
unknown         а
,__inference_sequential_5_layer_call_fn_63731p465%&Cб@
9б6
,і)
embedding_5_input         У
p 

 
ф "!і
unknown         ┐
#__inference_signature_wrapper_63804Ќ465%&PбM
б 
FфC
A
embedding_5_input,і)
embedding_5_input         У";ф8
6
activation_5&і#
activation_5         