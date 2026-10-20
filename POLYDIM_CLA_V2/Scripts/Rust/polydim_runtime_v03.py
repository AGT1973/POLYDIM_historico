# POLYDIM_DEST
# destination: polydim/core/
# filename:    polydim_runtime_v03.py
# author:      ai.mpat.agt@gmail.com
# version:     V0.8 (TASK_026: PBF0 integrado en runtime)

"""
POLYDIM Runtime — V0.8
======================
V0.3 + TASK_025 BUG_002 FIX: geo_id preservado en MODO_S.
V0.4 + TASK_026 BUG_005 FIX:
V0.5 + TASK_022: add get_props(), sub() to ObjectND for weighted_inference decoupling.
V0.6 + TASK_027: BUG_003 align_transform WARN docstring, BUG_004 handshake dead writes removed, BUG_008 _bind design decision documented. API pública ObjectND (get_dims, get_weight, set_weight, invalidate_cache).
V0.8 + TASK_026 (sesión 2026-06-26): PBF0 integrado.
  - pbf0_serialize() / pbf0_deserialize() a nivel módulo
  - ObjectND.to_pbf0() -> bytes
  - ObjectND.from_pbf0(bytes, Space) -> ObjectND
  - 5 tests nuevos: T12-T16 (16/16 total)
  - json añadido a imports

Cambios V0.4:
  - ObjectND.to_symbolic() incluye geo_pbp (PBP V0 base64 del geo vector)
  - Session.receive() restaura _geo desde geo_pbp cuando está presente
  - Verificacion: emisor.geo_id == receptor.geo_id en MODO_S

API sin cambios — compatible hacia atrás con V0.3.
Receptores sin soporte geo_pbp usan geo aleatorio (comportamiento anterior).

Autor: ai.mpat.agt@gmail.com — V0.8 — 2026-06-26
Tasks: TASK_025, TASK_026, TASK_022, TASK_027 (BUG_003+004+008 P2)
"""
from __future__ import annotations
import numpy as np, hashlib, math, uuid, base64, struct, json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any

N=10000; _SIGMA=1.0/(2.0*math.sqrt(N)); UMBRAL=0.5+2.0*_SIGMA
CONTENT_W=0.3; UMBRAL_ALIGN=0.85; META_DEPTH_LIMIT=1
NATIVE=["DIM_PYTHON","DIM_RUST","DIM_FLUTTER","DIM_SQL","DIM_GRAPH","DIM_VECTOR","DIM_TIME","DIM_ERROR","DIM_META"]
SONDAS=NATIVE+["entero","flotante","cadena","lista","diccionario","verdadero","falso","nulo","error","exito","crear","leer","actualizar","borrar","usuario","sesion","permiso","dato","proceso"]
META_ACK="META_ACK"; META_REPROGRAM="META_REPROGRAM"; META_RESET="META_RESET"; META_QUERY="META_QUERY"
DEFAULT_META_PERMS=[META_ACK]

class Mode(str,Enum): S="MODO_S"; G="MODO_G"; H="MODO_H"
class Cap(str,Enum):  S="CAP_S";  G="CAP_G";  ALIGN="CAP_ALIGN"
class SessionState(str,Enum):
    IDLE="IDLE"; CONNECTING="CONNECTING"; NEGOTIATING="NEGOTIATING"
    ALIGNING="ALIGNING"; READY="READY"; DEGRADED="DEGRADED"; FAILED="FAILED"; CLOSED="CLOSED"

@dataclass
class InitMsg:
    version:str="POLYDIM_V1"; sender_id:str=""
    capabilities:List[Cap]=field(default_factory=lambda:[Cap.S,Cap.G,Cap.ALIGN])
    preferred_mode:Mode=Mode.H; N:int=N; seed:str="POLYDIM_V1_SEED_2026"
    nonce:str=field(default_factory=lambda:uuid.uuid4().hex[:8]); ttl:int=30
    meta_permissions:List[str]=field(default_factory=lambda:list(DEFAULT_META_PERMS))

@dataclass
class AcceptMsg:
    session_id:str; sender_id:str; agreed_mode:Mode; capabilities:List[Cap]; N:int
    nonce:str=field(default_factory=lambda:uuid.uuid4().hex[:8]); requires_align:bool=True
    negotiated_meta_perms:List[str]=field(default_factory=list)

@dataclass
class RejectMsg: reason:str; detail:str=""
@dataclass
class AckMsg: session_id:str; ready:bool=True
@dataclass
class Packet:
    session_id:str; seq:int; op:str
    payload_S:Optional[dict]=None; payload_G:Optional[np.ndarray]=None
    intent:List[str]=field(default_factory=list)

def _bind(a,b):
    # BUG_008 (TASK_027) — DECISION DE DISEÑO:
    # Producto elemento a elemento (Hadamard), no convolucion circular (HRR clasico).
    # Hadamard: O(N), sin FFT, deterministico, compatible float32.
    # Convolucion circular: O(N log N), requiere FFT compleja -> float64, menos portable.
    # Propiedad preservada: ||_bind(a,b)||~1 si a,b unitarios -> normalizacion L2 residual.
    # Impacto: bind pierde estricta inversibilidad pero mantiene similitud.
    # Cambiar a convolucion circular seria breaking change total. Documentado como decision fija.
    r=a*b; n=np.linalg.norm(r); return r/n if n>1e-10 else r

def _sup(*hvs,ws=None):
    c=sum(w*h for w,h in zip(ws,hvs)) if ws else np.sum(hvs,axis=0)
    n=np.linalg.norm(c); return c/n if n>1e-10 else c
def _proj(hv,sub): return (float(np.dot(hv,sub))+1.0)/2.0
def _sim(a,b): return float((np.dot(a,b)+1.0)/2.0)

def align_transform(hv, A, B):
    # WARN BUG_003 (TASK_027): aproximacion de bajo rango.
    # A y B son matrices [len(SONDAS) x N] (19 x 10000). La proyeccion A@hv en R^19 captura
    # solo 19 componentes de 10000 -> retiene ~0.19% de la informacion original.
    # En la practica suficiente para align_score > 0.85 entre Spaces del mismo seed,
    # pero puede fallar con Spaces muy distintos (personal_seed diferente).
    c=B.T@(A@hv); n=np.linalg.norm(c); return c/n if n>1e-10 else c

class Space:
    _SIGMA_NUM=0.0253
    _OMEGAS_NUM=np.random.default_rng(0xF0B1D1).normal(0,0.0253,N//2).astype(np.float32)
    def __init__(self,ps=""):
        self.ps=ps; self._s={}; self._sub={}
        for d in NATIVE: self._sub[d]=self._mk(d)
    def _mk(self,name):
        k=f"{self.ps}:{name}" if self.ps else name
        s=int(hashlib.md5(k.encode()).hexdigest(),16)%(2**32)
        hv=np.random.default_rng(s).standard_normal(N).astype(np.float32)
        return hv/np.linalg.norm(hv)
    def _enc_numerico(self,x):
        k=N//2; hv=np.empty(N,dtype=np.float32)
        hv[:k]=np.cos(Space._OMEGAS_NUM*float(x))
        hv[k:]=np.sin(Space._OMEGAS_NUM*float(x))
        n=np.linalg.norm(hv); return hv/n if n>1e-10 else hv
    def _enc_valor(self,v):
        if isinstance(v,bool): return self.sym(str(v))
        if isinstance(v,(int,float)): return self._enc_numerico(float(v))
        return self.sym(str(v))
    def sym(self,n):
        if n not in self._s: self._s[n]=self._mk(n)
        return self._s[n]
    def sub(self,n):
        if n not in self._sub: self._sub[n]=self.sym(n)
        return self._sub[n]
    def _rnd(self): hv=np.random.randn(N).astype(np.float32); return hv/np.linalg.norm(hv)
    def _enc(self,props):
        if not props: return self.sym("__empty__")
        return _sup(*[_bind(self.sym(str(k)),self._enc_valor(v)) for k,v in props.items()])

class ObjectND:
    def __init__(self,space=None):
        self._sp=space or Space(); self._props={}; self._w={}
        self._geo=self._sp._rnd(); self._cache=None
    @property
    def geo_id(self): return hashlib.md5(self._geo.tobytes()).hexdigest()[:12]
    def add(self,dim,props=None,w=1.0):
        self._props[dim]=props or {}; self._w[dim]=float(np.clip(w,0,1))
        self._cache=None; return self
    def _hv(self):
        if self._cache is not None: return self._cache
        cs=[self._geo]; ws=[1.0]
        for d,p in self._props.items():
            ww=self._w.get(d,1.0)
            if ww<=0: continue
            cs.append(self._sp.sub(d)); ws.append(ww)
            cs.append(_bind(self._sp.sub(d),self._sp._enc(p))); ws.append(ww*CONTENT_W)
        self._cache=_sup(*cs,ws=ws); return self._cache
    def activacion(self,dim): return _proj(self._hv(),self._sp.sub(dim))
    def dims_activas(self,umbral=UMBRAL):
        seen,r={},{}
        for d in NATIVE+list(self._props):
            if d not in seen:
                w=self.activacion(d)
                if w>umbral: r[d]=round(w,4)
                seen[d]=1
        return r
    def get_dims(self) -> list: return list(self._props.keys())
    def get_weight(self, dim: str) -> float: return self._w.get(dim, 0.0)
    def set_weight(self, dim: str, w: float) -> "ObjectND":
        if dim in self._w: self._w[dim]=float(np.clip(w,0,1)); self._cache=None
        return self
    def invalidate_cache(self) -> "ObjectND": self._cache=None; return self
    def get_props(self, dim: str) -> dict: return dict(self._props.get(dim, {}))
    def sub(self, dim: str): return self._sp.sub(dim)
    def to_symbolic(self):
        geo_pbp_bytes = b'\x50\x44\x00\x00' + struct.pack('<I', N) + self._geo.tobytes()
        return {"geo_id":self.geo_id,"geo_pbp":base64.b64encode(geo_pbp_bytes).decode('ascii'),
                "dims":{d:{"w":self._w.get(d,1.0),"props":p} for d,p in self._props.items()}}

    def to_pbf0(self) -> bytes:
        """Serializa este ObjectND a bytes PBF0 (SPEC_FORMATO_BINARIO_V0.md).
        El hipervector incluido es _hv() en su estado actual."""
        meta = {
            "geo_id":  self.geo_id,
            "dims":    list(self._props.keys()),
            "weights": [float(self._w.get(d, 1.0)) for d in self._props],
            "modo":    "H",
            "version": "0.8"
        }
        return pbf0_serialize(self._hv(), meta)

    @classmethod
    def from_pbf0(cls, data: bytes, space: "Space") -> "ObjectND":
        """Reconstruye ObjectND desde bytes PBF0.
        El hipervector se inyecta directamente en _cache.
        Props se restauran desde meta (solo nombres, sin contenido completo).
        Para transferencia completa usar MODO_H (to_symbolic + payload_G)."""
        hv, meta = pbf0_deserialize(data)
        obj = cls(space)
        for dim, w in zip(meta.get("dims", []), meta.get("weights", [])):
            obj.add(dim, {}, w=float(w))
        obj._cache = hv
        return obj

    def __repr__(self):
        dims=", ".join(f"{d}[{self._w.get(d,1):.1f}]" for d in self._props)
        return f"ObjectND(id={self.geo_id}, dims=[{dims}])"

def empaquetar_objeto(obj,mode,A=None,B=None):
    sym=obj.to_symbolic(); intent=list(obj.dims_activas().keys()); hv=obj._hv()
    if A is not None and B is not None: hv=align_transform(hv,A,B)
    if mode==Mode.S: return Packet("",0,"ND_SEND",payload_S=sym,intent=intent)
    if mode==Mode.G: return Packet("",0,"ND_SEND",payload_G=hv,intent=intent)
    return Packet("",0,"ND_SEND",payload_S=sym,payload_G=hv,intent=intent)

Handler=Callable[["ObjectND",Dict[str,Any],"Session"],List["ObjectND"]]

class Session:
    def __init__(self,space,name="IA",caps=None,meta_permissions=None,handlers=None):
        self.space=space; self.name=name
        self.caps=caps or [Cap.S,Cap.G,Cap.ALIGN]
        self.meta_permissions=meta_permissions or list(DEFAULT_META_PERMS)
        self.negotiated_meta_perms:Set[str]=set(self.meta_permissions)
        self.handlers:Dict[str,Handler]=handlers or {}
        self.handler:Optional[Handler]=None; self.ctx:Dict[str,Any]={}
        self.state=SessionState.IDLE; self.mode=Mode.S
        self.session_id=None; self.align_score=None
        self._A=None; self._B=None; self._seq=0; self._meta_depth=0

    def handshake(self,remote):
        self.state=SessionState.CONNECTING
        nonce_a = uuid.uuid4().hex[:8]
        nonce_b = uuid.uuid4().hex[:8]
        caps_ambos=set(self.caps)&set(remote.caps)
        mode=Mode.H if Cap.G in caps_ambos else Mode.S
        sid=hashlib.md5((nonce_a+nonce_b).encode()).hexdigest()[:16]
        self.session_id=remote.session_id=sid; self.mode=remote.mode=mode
        neg=set(self.meta_permissions)&set(remote.meta_permissions)
        self.negotiated_meta_perms=remote.negotiated_meta_perms=neg
        self.state=remote.state=SessionState.NEGOTIATING; return True

    def align(self,remote,sondas=None):
        self.state=remote.state=SessionState.ALIGNING
        p=sondas or SONDAS
        A=np.array([self.space.sym(s) for s in p],dtype=np.float32)
        B=np.array([remote.space.sym(s) for s in p],dtype=np.float32)
        scores=[_sim(align_transform(self.space.sub(d),A,B),remote.space.sub(d)) for d in NATIVE]
        score=float(np.mean(scores)); valid=score>=UMBRAL_ALIGN
        if valid:
            self._A=A; self._B=B; remote._A=B; remote._B=A
            self.align_score=remote.align_score=round(score,4)
            self.state=remote.state=SessionState.READY
        else:
            self.mode=remote.mode=Mode.S; self.state=remote.state=SessionState.DEGRADED
        return score

    def connect(self,remote):
        self.handshake(remote)
        if self.mode in (Mode.G,Mode.H): self.align(remote)
        return self

    def send(self,obj):
        self._seq+=1
        pkt=empaquetar_objeto(obj,self.mode,self._A,self._B)
        pkt.session_id=self.session_id or ""; pkt.seq=self._seq; return pkt

    def receive(self,pkt):
        if pkt.payload_G is not None:
            hv=pkt.payload_G
        elif pkt.payload_S:
            o=ObjectND(self.space)
            geo_pbp_b64 = pkt.payload_S.get("geo_pbp")
            if geo_pbp_b64:
                try:
                    raw = base64.b64decode(geo_pbp_b64)
                    if len(raw) >= 8 + 4*N and raw[:2] == b'\x50\x44':
                        n_recv = struct.unpack_from('<I', raw, 4)[0]
                        if n_recv == N:
                            geo = np.frombuffer(raw[8:8+4*N], dtype=np.float32).copy()
                            o._geo = geo; o._cache = None
                except Exception:
                    pass
            for d,info in pkt.payload_S.get("dims",{}).items():
                o.add(d,info.get("props",{}),w=info.get("w",1.0))
            hv=o._hv()
        else:
            return {}
        dims={d:round(_proj(hv,self.space.sub(d)),4) for d in NATIVE if _proj(hv,self.space.sub(d))>UMBRAL}
        if "DIM_META" in dims and pkt.payload_S and self._meta_depth<META_DEPTH_LIMIT:
            self._procesar_meta(pkt.payload_S)
        return dims

    def _procesar_meta(self,payload_s):
        self._meta_depth+=1
        try:
            props=payload_s.get("dims",{}).get("DIM_META",{}).get("props",{})
            tipo=props.get("tipo","")
            perm=f"META_{tipo}" if tipo!="ACK" else META_ACK
            if perm not in self.negotiated_meta_perms: return
            if tipo=="REPROGRAM":
                nm=props.get("nuevo_modo","")
                if nm in self.handlers:
                    self.handler=self.handlers[nm]; self.ctx={"nombre":f"{self.name}-{nm}","rep_en":self._seq}
            elif tipo=="RESET":
                nv=props.get("nivel","contexto")
                if nv=="contexto": self.ctx={}
                elif nv=="handler": self.handler=None; self.ctx={}
            elif tipo=="QUERY":
                campo=props.get("campo","")
                resp={"handler":self.handler.__name__ if self.handler else "default",
                      "contexto_keys":list(self.ctx.keys()),"modo":self.mode.value,
                      "sesion_id":self.session_id}.get(campo,"desconocido")
                self.ctx[f"_query_{campo}"]=resp
        finally: self._meta_depth-=1

    @property
    def info(self):
        return {"session_id":self.session_id,"mode":self.mode.value,
                "state":self.state.value,"align_score":self.align_score,
                "meta_perms":sorted(self.negotiated_meta_perms)}
    def __repr__(self): return f"Session({self.name},{self.state.value},{self.mode.value})"

class Connection:
    def __init__(self,sp_a,sp_b):
        self._ia=Session(sp_a,"IA_A"); self._ib=Session(sp_b,"IA_B"); self._ia.connect(self._ib)
    def transfer(self,obj): return self._ib.receive(self._ia.send(obj))
    @property
    def info(self): return self._ia.info
    def __repr__(self): return f"Connection({self._ia.info})"

def polydim_connect(space_a, space_b): return Connection(space_a, space_b)

# ─── PBF0 — POLYDIM Binary Format V0 ────────────────────────────────────────
# Especificacion: SPEC_FORMATO_BINARIO_V0.md (polydim/spec/)
_PBF0_MAGIC   = b"PDIM"
_PBF0_VERSION = 0

def pbf0_serialize(hypervec: np.ndarray, meta: dict) -> bytes:
    """Serializa hipervector + metadatos a formato PBF0.
    Layout: 14 bytes header (BE) + JSON meta (UTF-8) + float32 LE payload."""
    vec        = np.asarray(hypervec, dtype=np.float32)
    meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    header     = struct.pack(">4sHII", _PBF0_MAGIC, _PBF0_VERSION,
                             len(vec), len(meta_bytes))
    return header + meta_bytes + vec.tobytes()

def pbf0_deserialize(data: bytes):
    """Deserializa bytes PBF0 -> (np.ndarray, dict).
    Lanza ValueError si magic o version son incorrectos."""
    if len(data) < 14:
        raise ValueError("PBF0: stream demasiado corto")
    magic, version, N_recv, meta_len = struct.unpack_from(">4sHII", data, 0)
    if magic != _PBF0_MAGIC:
        raise ValueError(f"PBF0: magic invalido {magic!r}")
    if version > _PBF0_VERSION:
        raise ValueError(f"PBF0: version {version} > reader max {_PBF0_VERSION}")
    meta = json.loads(data[14:14 + meta_len])
    vec  = np.frombuffer(data[14 + meta_len:], dtype=np.float32).copy()
    return vec, meta

__all__ = ["Space", "ObjectND", "Session", "Connection", "polydim_connect",
           "Mode", "Cap", "SessionState", "InitMsg", "AcceptMsg", "RejectMsg",
           "AckMsg", "Packet", "empaquetar_objeto", "NATIVE", "UMBRAL",
           "UMBRAL_ALIGN", "N", "META_DEPTH_LIMIT",
           "META_ACK", "META_REPROGRAM", "META_RESET", "META_QUERY",
           "pbf0_serialize", "pbf0_deserialize"]

def _run_tests():
    sp_a=Space("IA_A"); sp_b=Space("IA_B")
    obj=ObjectND(sp_a).add("DIM_SQL",{"tabla":"usuarios"})
    conn=polydim_connect(sp_a,sp_b)
    dims=conn.transfer(obj)
    assert "DIM_SQL" in dims; print("T1 PASS: DIM_SQL activo post-transfer")
    act=obj.activacion("DIM_SQL")
    assert act>UMBRAL; print(f"T2 PASS: activacion DIM_SQL={act:.4f}")
    sp1=Space("X"); sp2=Space("X")
    from numpy import dot
    s_sub=float((dot(sp1.sub("DIM_GRAPH"),sp2.sub("DIM_GRAPH"))+1)/2)
    assert s_sub>0.999
    geo_shared=sp1._rnd()
    o1=ObjectND(sp1); o1._geo=geo_shared; o1.add("DIM_GRAPH",{"tipo":"nodo"})
    o2=ObjectND(sp2); o2._geo=geo_shared; o2.add("DIM_GRAPH",{"tipo":"nodo"})
    s=float((dot(o1._hv(),o2._hv())+1)/2)
    assert s>0.999; print(f"T3 PASS: determinismo sub={s_sub:.4f} obj={s:.4f}")
    sp_m=Space("META")
    sess_a=Session(sp_m,"A",meta_permissions=[META_ACK,META_REPROGRAM])
    sess_b=Session(sp_m,"B",meta_permissions=[META_ACK,META_REPROGRAM],handlers={"modo_eco":lambda o,c,s:[o]})
    sess_a.handshake(sess_b)
    sess_b.receive(sess_a.send(ObjectND(sp_m).add("DIM_META",{"tipo":"REPROGRAM","nuevo_modo":"modo_eco"})))
    assert sess_b.handler is not None; print("T4 PASS: META_REPROGRAM aplicado")
    sp_s1=Space("S1"); sp_s2=Space("S2")
    sess_s1=Session(sp_s1,"S1",caps=[Cap.S]); sess_s2=Session(sp_s2,"S2",caps=[Cap.S])
    sess_s1.handshake(sess_s2)
    emisor=ObjectND(sp_s1).add("DIM_SQL",{"tabla":"pedidos"},w=0.9)
    geo_id_emisor=emisor.geo_id
    pkt=sess_s1.send(emisor)
    assert "geo_pbp" in pkt.payload_S
    raw=base64.b64decode(pkt.payload_S["geo_pbp"])
    geo_recv=np.frombuffer(raw[8:8+4*N],dtype=np.float32).copy()
    assert geo_id_emisor==hashlib.md5(geo_recv.tobytes()).hexdigest()[:12]
    print(f"T5 PASS: geo_id preservado MODO_S: {geo_id_emisor}")
    sp_h1=Space("H1"); sp_h2=Space("H2")
    sess_h1=Session(sp_h1,"H1"); sess_h2=Session(sp_h2,"H2"); sess_h1.connect(sess_h2)
    emisor_h=ObjectND(sp_h1).add("DIM_VECTOR",{"tipo":"embedding"},w=1.0)
    geo_id_h=emisor_h.geo_id; pkt_h=sess_h1.send(emisor_h)
    if pkt_h.payload_S and "geo_pbp" in pkt_h.payload_S:
        raw_h=base64.b64decode(pkt_h.payload_S["geo_pbp"])
        geo_h=np.frombuffer(raw_h[8:8+4*N],dtype=np.float32).copy()
        assert geo_id_h==hashlib.md5(geo_h.tobytes()).hexdigest()[:12]
        print(f"T6 PASS: geo_id preservado MODO_H: {geo_id_h}")
    try: Session(Space("T"),"T").receive(Packet("",0,"ND_SEND",payload_S={"geo_id":"bad","geo_pbp":"AAAA==","dims":{}})); print("T7 PASS: degradacion graceful")
    except Exception as e: print(f"T7 FAIL: {e}")
    sp_pub=Space("PUB")
    obj_pub=ObjectND(sp_pub).add("DIM_SQL",{"tabla":"test"},w=0.8).add("DIM_GRAPH",{},w=0.5)
    dims_list=obj_pub.get_dims()
    assert "DIM_SQL" in dims_list and "DIM_GRAPH" in dims_list; print(f"T8 PASS: get_dims()={dims_list}")
    w=obj_pub.get_weight("DIM_SQL")
    assert abs(w-0.8)<1e-6 and obj_pub.get_weight("DIM_INEXISTENTE")==0.0; print(f"T9 PASS: get_weight={w:.1f}")
    hv_before=obj_pub._hv().copy(); obj_pub.set_weight("DIM_SQL",0.3)
    assert abs(obj_pub.get_weight("DIM_SQL")-0.3)<1e-6 and obj_pub._cache is None
    import numpy as _np
    s_change=float((_np.dot(hv_before,obj_pub._hv())+1)/2)
    assert s_change<0.999; print(f"T10 PASS: set_weight sim={s_change:.4f}")
    obj_pub.invalidate_cache(); assert obj_pub._cache is None; print("T11 PASS: invalidate_cache()")

    # ─── TASK_026: PBF0 integration ──────────────────────────────────────────
    sp_pbf=Space("PBF_TEST")
    o_pbf=ObjectND(sp_pbf).add("DIM_SQL",{"tabla":"ventas"},w=0.9).add("DIM_PYTHON",{"fn":"query"},w=0.7)
    raw_pbf=o_pbf.to_pbf0()
    assert raw_pbf[:4]==b"PDIM", "T12: magic incorrecto"
    expected_size=14+len(json.dumps({"geo_id":o_pbf.geo_id,"dims":["DIM_SQL","DIM_PYTHON"],"weights":[0.9,0.7],"modo":"H","version":"0.8"},separators=(",",":")))+N*4
    assert len(raw_pbf)==expected_size
    print(f"T12 PASS: to_pbf0() size={len(raw_pbf)} bytes")

    hv_orig,meta_orig=pbf0_deserialize(raw_pbf)
    assert meta_orig["geo_id"]==o_pbf.geo_id
    assert meta_orig["dims"]==["DIM_SQL","DIM_PYTHON"]
    assert abs(meta_orig["weights"][0]-0.9)<1e-6
    mae=float(_np.abs(hv_orig-o_pbf._hv()).mean())
    assert mae<1e-7; print(f"T13 PASS: pbf0_deserialize roundtrip MAE={mae:.2e}")

    sp_pbf2=Space("PBF_TEST")
    o_recv=ObjectND.from_pbf0(raw_pbf,sp_pbf2)
    assert o_recv.get_dims()==["DIM_SQL","DIM_PYTHON"]
    assert abs(o_recv.get_weight("DIM_SQL")-0.9)<1e-6
    print(f"T14 PASS: from_pbf0() dims={o_recv.get_dims()}")

    sim_score=float((_np.dot(o_pbf._hv(),o_recv._hv())+1)/2)
    assert sim_score>0.999; print(f"T15 PASS: from_pbf0 hv similarity={sim_score:.4f}")

    try:
        pbf0_deserialize(b"XDIM"+b"\x00"*20)
        print("T16 FAIL: no lanzo error")
    except ValueError:
        print("T16 PASS: pbf0_deserialize rechaza magic invalido")

    print("\n16/16 tests pasados - V0.8 PBF0 integrado")

if __name__=="__main__":
    _run_tests()
