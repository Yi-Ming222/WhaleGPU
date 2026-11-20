"""Deterministic teaching traces, not hardware scheduling observations."""
from dataclasses import dataclass

OPS={'SUM':(0,lambda a,b:a+b),'MAX':(float('-inf'),max),'MIN':(float('inf'),min)}
@dataclass(frozen=True)
class Stage:
    title:str
    before:tuple
    after:tuple
    offset:int=0
    edges:tuple=()
    shared:tuple=()
    block:int=0
    boundary:str=''
    mask:str='—'


def trace(values, mode='shared', op='SUM', block=0):
    if op not in OPS or mode not in ('shared','shuffle','block'):raise ValueError('Unknown mode/operator')
    if not values or len(values)>256:raise ValueError('Expected 1..256 values')
    identity,combine=OPS[op]
    size=(1<<(len(values)-1).bit_length()) if mode=='shared' else ((len(values)+31)//32)*32
    if mode=='shuffle' and size!=32:raise ValueError('Shuffle demonstration needs <=32 inputs')
    data=list(values)+[identity]*(size-len(values))
    states=[Stage(f'加载 {len(values)} 个有效值；填充 {size-len(values)} 个单位元',tuple(data),tuple(data),shared=tuple(data) if mode=='shared' else (),block=block,mask='0xffffffff' if mode!='shared' else '—')]
    def emit(title,before,edges=(),offset=0,shared=(),boundary='',mask='—'):
        states.append(Stage(title,tuple(before),tuple(data),offset,tuple(edges),tuple(shared),block,boundary,mask))
    if mode=='shared':
        emit('共享内存加载完成 / block 屏障',data,shared=data,boundary='__syncthreads(): 仅当前 Block')
        offset=size//2
        while offset:
            before=data[:];edges=[(i,i+offset) for i in range(offset)]
            for i,j in edges:data[i]=combine(before[i],before[j])
            emit('Shared Memory 读取 → 合并 → 写回',before,edges,offset,data)
            emit('本轮写回完成 / block 屏障',data,offset=offset,shared=data,boundary='__syncthreads(): 仅当前 Block')
            offset//=2
    else:
        for offset in (16,8,4,2,1):
            before=data[:];edges=[(i,i+offset) for i in range(size) if i%32+offset<32]
            for i,j in edges:data[i]=combine(before[i],before[j])
            emit('Warp 内 shuffle-down + 有效来源合并',before,edges,offset,mask='0xffffffff')
        if mode=='block':
            partial=data[::32]
            emit('各 Warp Lane 0 → Shared Memory[warpId]',data,shared=partial)
            emit('等待所有 Warp 写入完成',data,shared=partial,boundary='__syncthreads(): 仅当前 Block')
            data=partial+[identity]*(32-len(partial))
            emit('Warp 0 加载局部和，其余 Lane 填充单位元',data,shared=partial,mask='0xffffffff')
            for offset in (16,8,4,2,1):
                before=data[:];edges=[(i,i+offset) for i in range(32-offset)]
                for i,j in edges:data[i]=combine(before[i],before[j])
                emit('Warp 0 合并局部结果',before,edges,offset,partial,mask='0xffffffff')
    return states


def two_stage(values,op='SUM',block_size=256):
    if not 1<=block_size<=256 or not values:raise ValueError('Invalid input')
    states=[];partials=[]
    for b,start in enumerate(range(0,len(values),block_size)):
        local=trace(values[start:start+block_size],'block',op,b)
        states.extend(local);partials.append(local[-1].after[0])
        states.append(Stage('Kernel 1: 写入 global partial[blockId]',tuple(partials),tuple(partials),block=b,boundary='各 Block 独立；展示顺序不是硬件调度顺序'))
    if len(partials)>256:raise ValueError('Teaching view supports <=256 blocks')
    states.append(Stage('Kernel 1 完成 → Kernel 2 开始',tuple(partials),tuple(partials),boundary='同一 stream 的 Kernel 边界；不是 __syncthreads()'))
    states.extend(trace(partials,'block',op))
    return states
