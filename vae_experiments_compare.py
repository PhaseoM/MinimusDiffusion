r"""Compare VAE ablations with paper-style reconstruction and latent metrics.

    python vae_experiments_compare.py --device cuda:3
    python vae_experiments_compare.py --device cuda:3 --max-samples 256
    python vae_experiments_compare.py --device cuda:3 --output-dir results/my_comparison
    python vae_experiments_compare.py --from-report results/existing_comparison.html
    python vae_experiments_compare.py --device cuda:3 \
        --base-results results/vae_three_experiments_comparison_20261030/metrics.json \
        --append-run results/VAE_20261030_165055

Ordered experiments: (z_channels=32, beta=.5), (8, .5), (8, .01).
Use --append-run to add C8 beta ablations. --base-results reuses compatible
evaluations after checking image IDs, settings, metric hashes and sample scores.
The same frozen epoch (default 100) and training data split are required.
Both posterior-sampled and posterior-mean reconstruction are evaluated. Metrics
include reconstruction FID (official pytorch-fid 2048-D FID Inception), LPIPS
(AlexNet v0.1, authors' code and pretrained weights), PSNR, SSIM, MSE, KL and
aggregate-posterior diagnostics. Dependencies are the project's torch,
torchvision, numpy, pillow and standard library; no package is installed.

Upstream metric code and weights are fetched from their official repositories
directly into RAM, with SHA256 recorded. This needs internet access; no pretrained
cache is written. Only a newly created output directory may receive writes.
Output: report.html, summary.csv, per_image_metrics.csv, metrics.json,
comparison_table.tex, reconstructions_sample.png, reconstructions_mean.png.
Full test N=3000 is preferred; subset rFID has substantial finite-sample bias.
"""


def interpretation_appendix(payload):
    """Describe measured reconstruction/prior trade-offs without equating KL to MI."""
    a, b, c = payload["experiments"][:3]
    lb, lc = b["latent"], c["latent"]
    eb = b["modes"]["sample"]["negative_elbo_nats_per_image_mc"]
    ec = c["modes"]["sample"]["negative_elbo_nats_per_image_mc"]
    z_delta = payload["paired_comparisons"]["z_channels_32_to_8"]["modes"]["sample"]["rfid_delta"]
    z_base = a["modes"]["sample"]["rfid_2048"]
    direction = "升高（标准 ELBO 变差）" if ec > eb else "降低（标准 ELBO 改善）"
    return f"""<div class="card" id="prior-tradeoff"><p><strong>重建与先验约束的实测折中：</strong>
    B→C 的聚合标准差从 {lb['aggregate_pooled_std']:.4f} 变为 {lc['aggregate_pooled_std']:.4f}（标准正态目标为 1），
    逐坐标高斯矩匹配 KL 代理从 {lb['aggregate_diagonal_gaussian_moment_kl_nats_per_latent']:.5f} 变为
    {lc['aggregate_diagonal_gaussian_moment_kl_nats_per_latent']:.5f} nats/latent，
    最大通道相关系数绝对值从 {lb['channel_correlation_max_abs']:.4f} 变为 {lc['channel_correlation_max_abs']:.4f}。
    这些具体诊断显示前两阶矩及相关性的先验匹配变差，而非仅根据条件 KL 推断真实聚合 KL。</p>
    <p>标准负 ELBO 从 {eb:.1f} 变为 {ec:.1f} nats/image，{direction}。
    因此本次重建指标改善不等于标准概率目标也改善；仍需区分 β 加权训练目标、标准 ELBO 与下游扩散效果。</p>
    <p>A→B 的采样 rFID 仅变化 {z_delta:+.5f}（相对 {100*z_delta/z_base:+.4f}%）。
    该量级不能按小数末位直接宣称感知质量优劣；本次没有估计 rFID 的重复训练或 bootstrap 不确定性。</p>
    <p>如果后续将这些表示用于 LDM，应在训练集上重新估计 latent scaling，并在解码时做对应反变换；
    此处测试集标准差只用于诊断，不用于拟合训练预处理。
    标量缩放不会消除通道相关性和非高斯结构，且潜分布偏离标准正态并不等于扩散模型无法学习。
    标准差缩放可参考 <a href="https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddpm.py">LDM 官方 scale_by_std 实现</a>。</p></div>"""


def beta_sweep_appendix(payload):
    """Add the intermediate-beta comparison when additional C8 runs are present."""
    experiments = payload["experiments"]
    if len(experiments) <= 3:
        return ""
    sweep = sorted((r for r in experiments if r["z_channels"] == 8), key=lambda r: r["beta"], reverse=True)
    rows = []
    for r in sweep:
        m, latent = r["modes"]["sample"], r["latent"]
        rows.append(
            f"<tr><td>{r['label'].split(':')[0]}</td><td>{r['beta']:g}</td>"
            f"<td>{m['rfid_2048']:.4f}</td><td>{m['psnr_db']['mean']:.3f}</td>"
            f"<td>{m['ssim']['mean']:.4f}</td><td>{m['lpips_alex_v01']['mean']:.4f}</td>"
            f"<td>{latent['conditional_kl_nats_per_latent']:.5f}</td>"
            f"<td>{latent['aggregate_pooled_std']:.5f}</td>"
            f"<td>{latent['aggregate_diagonal_gaussian_moment_kl_nats_per_latent']:.5f}</td>"
            f"<td>{latent['channel_correlation_max_abs']:.5f}</td></tr>"
        )
    descriptions = []
    for added in experiments[3:]:
        label = added["label"].split(":")[0]
        for reference in experiments[1:3]:
            for mode in ["sample", "mean"]:
                a, b = reference["modes"][mode], added["modes"][mode]
                descriptions.append(
                    f"<p>{label}（β={added['beta']:g}）相对 {reference['label'].split(':')[0]}"
                    f"（β={reference['beta']:g}），{mode} 模式："
                    f"PSNR {b['psnr_db']['mean']-a['psnr_db']['mean']:+.3f} dB，"
                    f"SSIM {b['ssim']['mean']-a['ssim']['mean']:+.4f}，"
                    f"LPIPS {b['lpips_alex_v01']['mean']-a['lpips_alex_v01']['mean']:+.4f}，"
                    f"rFID {b['rfid_2048']-a['rfid_2048']:+.4f}，"
                    f"MSE 相对变化 {100*(b['mse_01']['mean']/a['mse_01']['mean']-1):+.2f}%。</p>"
                )
        l, ref_l = added["latent"], experiments[2]["latent"]
        descriptions.append(
            f"<p>{label} 的后验噪声方差为 {l['posterior_z_variance']:.6f}，"
            f"聚合标准差为 {l['aggregate_pooled_std']:.5f}，逐坐标高斯 KL 代理为 "
            f"{l['aggregate_diagonal_gaussian_moment_kl_nats_per_latent']:.5f} nats/latent，"
            f"最大通道相关绝对值为 {l['channel_correlation_max_abs']:.5f}。"
            f"对照 C（β=0.01）的后三项：{ref_l['aggregate_pooled_std']:.5f}、"
            f"{ref_l['aggregate_diagonal_gaussian_moment_kl_nats_per_latent']:.5f}、"
            f"{ref_l['channel_correlation_max_abs']:.5f}。这些实测值与重建表共同描述工作点，"
            "不能只按 β 大小预设最优结论。</p>"
        )
    return (
        '<section id="beta-sweep"><h2>新增实验：固定 C8 的 β 扫描</h2>'
        '<div class="scroll"><table><tr><th>实验</th><th>β</th><th>rFID ↓</th><th>PSNR ↑</th>'
        '<th>SSIM ↑</th><th>LPIPS ↓</th><th>KL / latent</th><th>聚合 std</th>'
        '<th>高斯 KL 代理 / latent</th><th>最大 |corr|</th></tr>'
        + ''.join(rows) + '</table></div>' + ''.join(descriptions)
        + '<p>该表把同一架构的 β 按降序排列。聚合 std 的标准正态目标为 1，'
        '高斯矩匹配代理与相关性仅为先验匹配诊断；最终 LDM 效果仍需单独验证。'
        '新增配对差及其测试样本区间列于下一节。</p></section>'
    )


def export_comparison_report(input_report, output_dir=None):
    """Export an existing embedded report without downloading or rerunning models."""
    import base64
    import csv
    import hashlib
    import json
    import re
    from pathlib import Path

    source = Path(input_report).resolve()
    document = source.read_text(encoding="utf-8")
    match = re.search(r'<script id="comparison-data" type="application/json">(.*?)</script>', document, re.S)
    if match is None:
        raise ValueError("The input HTML does not contain comparison-data")
    payload = json.loads(match.group(1))
    destination = Path(output_dir).resolve() if output_dir else source.parent / source.stem
    destination.mkdir(parents=True, exist_ok=False)

    def download_content(name):
        pattern = r'download="' + re.escape(name) + r'"[^>]*href="data:[^;]+;base64,([^"]+)"'
        found = re.search(pattern, document)
        if found is None:
            raise ValueError(f"Missing embedded download: {name}")
        return base64.b64decode(found.group(1))

    document = document.replace(
        "所有新增内容保存在这个 HTML 内，预训练权重和评估源码只载入内存，未安装依赖、未写权重缓存、未导出其他文件。",
        "本报告嵌入全部图像与数据，同目录另行导出 CSV、JSON、LaTeX 和 PNG。预训练权重和评估源码只载入内存，未安装依赖或写权重缓存。",
    )
    document = document.replace(
        "这两项共同描述编码信息量与注入噪声的变化；",
        "这两项反映逐样本后验约束与注入噪声的变化；条件 KL 同时包含互信息和聚合先验失配，不能直接当作有效信息量；",
    )
    if 'id="prior-tradeoff"' not in document:
        document = document.replace("<h2>7.", interpretation_appendix(payload) + "<h2>7.", 1)
    if 'id="beta-sweep"' not in document and len(payload["experiments"]) > 3:
        document = document.replace("<h2>7.", beta_sweep_appendix(payload) + "<h2>7.", 1)
    (destination / "report.html").write_text(document, encoding="utf-8")
    (destination / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (destination / "summary.csv").write_bytes(download_content("vae_comparison.csv"))
    (destination / "comparison_table.tex").write_bytes(download_content("vae_comparison.tex"))
    fields = ["experiment", "mode", "image_id", "conditional_kl_nats", "mse_01", "mae_01", "psnr_db", "ssim", "lpips_alex_v01", "mse_raw_m11", "clipped_pixel_fraction"]
    with (destination / "per_image_metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for experiment in payload["experiments"]:
            for mode, values in experiment["per_image"].items():
                for index, image_id in enumerate(payload["protocol"]["test_image_ids"]):
                    writer.writerow({
                        "experiment": experiment["label"], "mode": mode, "image_id": image_id,
                        "conditional_kl_nats": experiment["per_image_conditional_kl_nats"][index],
                        **{key: values[key][index] for key in fields[4:]},
                    })
    for mode in ["sample", "mean"]:
        image = re.search(r'id="' + mode + r'-grid"[^>]*src="data:image/png;base64,([^"]+)"', document)
        if image is None:
            raise ValueError(f"Missing {mode} reconstruction figure")
        (destination / f"reconstructions_{mode}.png").write_bytes(base64.b64decode(image.group(1)))
    manifest = {
        "source_report": str(source), "source_report_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "export_script": str(Path(__file__).resolve()), "export_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": "Existing evaluated data exported without recomputing metrics; computation protocol is in metrics.json.",
    }
    (destination / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported report: {destination / 'report.html'}", flush=True)
    return destination


def main():

    import argparse
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dirs",nargs=3,default=[
        "results/VAE_20261029_1131","results/VAE_20261030_000704","results/VAE_20261030_064739"],
        metavar=("C32_B05","C8_B05","C8_B001"),help="Ordered run directories for the three controlled ablations")
    parser.add_argument("--epoch",type=int,default=100)
    parser.add_argument("--device",default="auto")
    parser.add_argument("--max-samples",type=int,help="Default: all 3000 test images; a subset makes rFID less reliable")
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--seed",type=int,default=20261030)
    parser.add_argument("--output-dir",help="A NEW output directory; existing directories are refused")
    parser.add_argument("--from-report",help="Export an existing comparison HTML without recomputing metrics")
    parser.add_argument("--append-run",action="append",default=[],help="Additional C8 beta-ablation run; repeat to add multiple runs")
    parser.add_argument("--base-results",help="Existing metrics.json for compatible evaluated runs; new runs are evaluated in full")
    args=parser.parse_args()
    if args.epoch<=0 or args.batch_size<=0 or args.seed<0 or (args.max_samples is not None and args.max_samples<2):
        parser.error("epoch/batch-size must be positive, seed nonnegative, max-samples at least 2")
    if args.from_report and (args.append_run or args.base_results):
        parser.error("--from-report cannot be combined with evaluation/cache options")
    if len(args.append_run)>23:
        parser.error("At most 23 additional runs can be labelled D through Z")
    if args.from_report:
        export_comparison_report(args.from_report,args.output_dir)
        return

    import os,sys,io,json,math,hashlib,base64,csv,time,re,types,urllib.request,warnings
    from pathlib import Path
    from datetime import datetime
    os.environ["PYTHONDONTWRITEBYTECODE"]="1"
    os.environ["CUDA_CACHE_DISABLE"]="1"
    os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"]="1"
    sys.dont_write_bytecode=True
    import tempfile
    tempfile.tempdir="/tmp"
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchvision
    from PIL import Image
    from torch.utils.data import DataLoader,Subset
    from vae_eval import _ImageDataset,_image_transform,_ssim

    ROOT=Path(__file__).resolve().parent
    # Load the unchanged VAE source without unrelated model-package initializers.
    # The private namespace preserves relative imports and leaves training imports alone.
    for suffix,relative in [("", "models"),(".modules", "models/modules"),(".autoencoders", "models/autoencoders")]:
        name="_vae_comparison_models"+suffix
        package=types.ModuleType(name)
        package.__path__=[str(ROOT/relative)]
        package.__package__=name
        sys.modules[name]=package
    from _vae_comparison_models.autoencoders.vae_midattn import VAE

    OUTPUT_DIR=Path(args.output_dir).resolve() if args.output_dir else ROOT/"results"/f"vae_experiments_comparison_{datetime.now():%Y%m%d_%H%M%S_%f}"
    OUTPUT_DIR.mkdir(parents=True,exist_ok=False)
    OUTPUT=OUTPUT_DIR/"report.html"
    # Allow writes only inside the newly created results directory.
    def audit(event,args):
        if event=="open":
            target,mode,flags=args
            writing=(isinstance(mode,str) and any(c in mode for c in "wax+")) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if writing and not (isinstance(target,(str,bytes,os.PathLike)) and Path(target).resolve().is_relative_to(OUTPUT_DIR)):
                raise PermissionError(f"Only new result files may be written: {target!r}")
        elif event in ("os.mkdir","os.remove","os.rmdir","os.rename","os.link","os.symlink","os.truncate"):
            raise PermissionError(f"Filesystem mutation disabled: {event}")
    sys.addaudithook(audit)

    DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else torch.device(args.device)
    if DEVICE.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use an accessible GPU or --device cpu")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    SEED=args.seed
    BATCH=args.batch_size
    RUNS=args.run_dirs+args.append_run
    LABELS=["A: C32 beta0.5","B: C8 beta0.5","C: C8 beta0.01"]
    cached_payload=None
    cache_index={}
    cache_checks={}
    if args.base_results:
        cached_payload=json.loads(Path(args.base_results).read_text(encoding="utf-8"))
        cache_index={str(Path(r["checkpoint"]).resolve()):r for r in cached_payload["experiments"]}
        if cached_payload["protocol"]["torch"]!=torch.__version__ or cached_payload["protocol"]["torchvision"]!=torchvision.__version__:
            raise ValueError("Cached metric library versions differ; recompute instead of reusing")
        if cached_payload["protocol"]["posterior_seed_base"]!=SEED:
            raise ValueError("Cached posterior seed differs")
        if cached_payload["protocol"]["batch_size"]!=BATCH:
            raise ValueError("Use the same batch size as the cached protocol")
    for index,run in enumerate(args.append_run):
        path=Path(run)
        if not path.is_absolute():path=ROOT/path
        ck=torch.load(path/"ckpts"/f"ckpt_epoch_{args.epoch:04d}.pth",map_location="cpu",weights_only=True,mmap=True)
        if ck["model_kwargs"]["z_channels"]!=8:
            raise ValueError("--append-run currently supports the C8 beta-ablation family")
        LABELS.append(f"{chr(68+index)}: C8 beta{ck['config']['model']['beta']:g}")
        del ck
    download_manifest={}
    def fetch(url,expected_prefix=None):
        print("[fetch]",url,flush=True)
        request=urllib.request.Request(url,headers={"User-Agent":"MinimusDiffusion-readonly-evaluation"})
        for attempt in range(1,4):
            try:
                # Retry the official HTTPS URL directly when the configured proxy fails.
                direct=(attempt%2==0)
                if url.startswith("https://download.pytorch.org/"):
                    direct=not direct
                opener=urllib.request.build_opener(urllib.request.ProxyHandler({})) if direct else urllib.request.build_opener()
                with opener.open(request,timeout=45) as response:
                    data=response.read()
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                print(f"[fetch retry] attempt {attempt}/3 ({'direct' if direct else 'configured proxy'}): {type(error).__name__}",flush=True)
                if attempt==3:
                    raise
                time.sleep(3)
        digest=hashlib.sha256(data).hexdigest()
        if expected_prefix and not digest.startswith(expected_prefix):
            raise ValueError(f"Weight checksum mismatch for {url}: {digest}, bytes={len(data)}")
        if cached_payload is not None and url in cached_payload["downloads"]:
            if digest!=cached_payload["downloads"][url]["sha256"]:
                raise ValueError(f"Metric source/weight differs from cached evaluation: {url}")
        download_manifest[url]={"sha256":digest,"bytes":len(data)}
        print("[fetch done]",len(data),digest[:16],flush=True)
        return data
    def tensor_state(data):
        return torch.load(io.BytesIO(data),map_location="cpu",weights_only=True)
    def ram_module(name,source,package=None):
        module=types.ModuleType(name)
        module.__file__="<RAM:"+name+">"
        module.__package__=package or name.rpartition(".")[0]
        sys.modules[name]=module
        exec(compile(source,module.__file__,"exec"),module.__dict__)
        return module

    # Use the authors' LPIPS modules, with both trunk and calibration weights loaded from RAM.
    lpips_root=types.ModuleType("lpips")
    lpips_root.__path__=[]
    lpips_root.normalize_tensor=lambda v,eps=1e-10:v/(torch.sqrt(torch.sum(v**2,dim=1,keepdim=True))+eps)
    sys.modules["lpips"]=lpips_root
    pn_source=fetch("https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/pretrained_networks.py")
    pn=ram_module("lpips.pretrained_networks",pn_source,"lpips")
    lpips_root.pretrained_networks=pn
    lp_source=fetch("https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/lpips.py")
    lp_module=ram_module("lpips.lpips",lp_source,"lpips")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lpips_model=lp_module.LPIPS(pretrained=False,net="alex",version="0.1",pnet_rand=True,eval_mode=True,verbose=False)
    alex_state=tensor_state(fetch("https://download.pytorch.org/models/alexnet-owt-7be5be79.pth","7be5be79"))
    for s in range(1,6):
        for name,layer in getattr(lpips_model.net,f"slice{s}").named_children():
            prefix=f"features.{name}."
            state={k[len(prefix):]:v for k,v in alex_state.items() if k.startswith(prefix)}
            layer.load_state_dict(state,strict=True)
    del alex_state
    calibration=tensor_state(fetch("https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/alex.pth"))
    lpips_model.load_state_dict(calibration,strict=False)
    for i in range(5):
        torch.testing.assert_close(lpips_model.lins[i].model[1].weight,calibration[f"lin{i}.model.1.weight"])
    lpips_model.pnet_rand=False
    lpips_model=lpips_model.to(DEVICE).eval().requires_grad_(False)

    # Official pytorch-fid feature extractor and TensorFlow-port FID weights.
    fid_source=fetch("https://raw.githubusercontent.com/mseitzer/pytorch-fid/master/src/pytorch_fid/inception.py")
    fid_module=ram_module("_readonly_fid_inception",fid_source)
    fid_bytes=fetch(fid_module.FID_WEIGHTS_URL,"6726825d")
    fid_weights=tensor_state(fid_bytes)
    del fid_bytes
    fid_module.load_state_dict_from_url=lambda *args,**kwargs:fid_weights
    fid_model=fid_module.InceptionV3([3],use_fid_inception=True).to(DEVICE).eval()
    del fid_weights

    # Read checkpoint metadata, including data protocols, instead of trusting current YAML paths.
    checkpoints=[]
    for label,run in zip(LABELS,RUNS):
        run_path=Path(run)
        if not run_path.is_absolute():run_path=ROOT/run_path
        path=run_path/"ckpts"/f"ckpt_epoch_{args.epoch:04d}.pth"
        ck=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
        checkpoints.append((label,path,ck))
    if len({str(path.resolve()) for _,path,_ in checkpoints})!=len(checkpoints):
        raise ValueError("Each experiment must use a distinct run/checkpoint")
    assert [(ck["model_kwargs"]["z_channels"],ck["config"]["model"]["beta"]) for _,_,ck in checkpoints[:3]]==[(32,0.5),(8,0.5),(8,0.01)],"Expected ordered base ablation: C32 beta0.5, C8 beta0.5, C8 beta0.01"
    base_config=checkpoints[0][2]["config"]
    for _,_,ck in checkpoints:
        for key in ["data","seed","batch_size","optim","train"]:
            assert ck["config"][key]==base_config[key],f"Uncontrolled config difference: {key}"
        assert ck["epoch"]==args.epoch-1
        assert ck["config"]["auto_resume"] is False
        assert ck["config"]["path"]["dataset_path"]==base_config["path"]["dataset_path"]
        for key,value in checkpoints[1][2]["model_kwargs"].items():
            if key!="z_channels":
                assert ck["model_kwargs"][key]==value,f"Uncontrolled model setting: {key}"

    dataset_root=ROOT/Path(base_config["path"]["dataset_path"])
    dataset=_ImageDataset(dataset_root,_image_transform(base_config["data"]["img_size"],3),3)
    n_val=int(len(dataset)*base_config["data"]["val_ratio"])
    n_train=len(dataset)-2*n_val
    permutation=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(base_config["seed"]))
    test_indices=permutation[n_train+n_val:].tolist()
    if args.max_samples is not None:test_indices=test_indices[:args.max_samples]
    N=len(test_indices)
    assert N>=2 and len(dataset)==30000
    image_ids=[int(dataset.paths[i].stem) for i in test_indices]
    if cached_payload is not None:
        cp=cached_payload["protocol"]
        if cp["test_image_ids"]!=image_ids or cp["test_indices"]!=test_indices:
            raise ValueError("Cached test images/order differ; use the same complete split or recompute")
        if cp["split_seed"]!=base_config["seed"] or cp["input_resize"]!=list(base_config["data"]["img_size"]):
            raise ValueError("Cached data preprocessing settings differ")
    loader=DataLoader(Subset(dataset,test_indices),batch_size=16,shuffle=False,num_workers=0)
    images=torch.empty(N,3,*base_config["data"]["img_size"],dtype=torch.float32)
    offset=0
    print("[dataset] loading fixed full test split",N,flush=True)
    for batch,_ in loader:
        images[offset:offset+len(batch)]=batch
        offset+=len(batch)
        if offset%256==0 or offset==N:print("[dataset]",offset,"/",N,flush=True)

    @torch.inference_mode()
    def features(x):
        return fid_model(x)[0].flatten(1)
    @torch.inference_mode()
    def paired_image_metrics(x,reconstruction):
        target=((x+1)*0.5).clamp(0,1)
        prediction=((reconstruction+1)*0.5).clamp(0,1)
        mse=(target-prediction).square().flatten(1).mean(1)
        return {
            "mse_01":mse,
            "mae_01":(target-prediction).abs().flatten(1).mean(1),
            "psnr_db":-10*mse.clamp_min(1e-12).log10(),
            "ssim":_ssim(target,prediction),
            "lpips_alex_v01":lpips_model(target*2-1,prediction*2-1).flatten(),
            "mse_raw_m11":(x-reconstruction).square().flatten(1).mean(1),
            "clipped_pixel_fraction":((reconstruction < -1)|(reconstruction > 1)).float().flatten(1).mean(1),
        },prediction

    # Checks meaningful to metric conventions before expensive evaluation.
    with torch.inference_mode():
        x=images[:2].to(DEVICE)
        t=(x+1)*0.5
        assert float(lpips_model(x,x).abs().max())<1e-7
        torch.testing.assert_close(_ssim(t,t),torch.ones(2,device=DEVICE),rtol=2e-5,atol=2e-5)
        torch.testing.assert_close(lpips_model(x,-x),lpips_model(-x,x),rtol=2e-5,atol=2e-6)
        f=features(t)
        assert f.shape==(2,2048) and torch.isfinite(f).all()
    print("[checks] pretrained weights, LPIPS identity/symmetry, SSIM identity, FID features passed",flush=True)

    real_features=[]
    with torch.inference_mode():
        for start in range(0,N,BATCH):
            x=((images[start:start+BATCH].to(DEVICE)+1)*0.5).clamp(0,1)
            real_features.append(features(x).cpu())
            if (start+len(x))%256==0 or start+len(x)==N:
                print("[reference features]",start+len(x),"/",N,flush=True)
    real_features=torch.cat(real_features)
    def gaussian_stats(feat):
        a=feat.to(device=DEVICE,dtype=torch.float64)
        m=a.mean(0)
        centered=a-m
        return m,centered.T@centered/(len(a)-1)
    ref_mean,ref_cov=gaussian_stats(real_features)
    eigenvalues,eigenvectors=torch.linalg.eigh(ref_cov)
    ref_sqrt=(eigenvectors*eigenvalues.clamp_min(0).sqrt()[None,:])@eigenvectors.T
    @torch.inference_mode()
    def rfid(feat):
        mean,cov=gaussian_stats(feat)
        product=ref_sqrt@cov@ref_sqrt
        product=(product+product.T)*0.5
        eig=torch.linalg.eigvalsh(product).clamp_min(0)
        result=((mean-ref_mean).square().sum()+ref_cov.trace()+cov.trace()-2*eig.sqrt().sum()).item()
        if result < -1e-5:raise ValueError(f"Negative rFID beyond rounding: {result}")
        return max(0.0,result)
    assert rfid(real_features)<1e-4
    print("[checks] rFID(X,X) < 1e-4 passed",flush=True)

    def summarize(values):
        out={}
        for key,rows in values.items():
            a=np.asarray(rows,dtype=np.float64)
            assert len(a)==N and np.isfinite(a).all(),key
            out[key]={"mean":float(a.mean()),"std_across_images":float(a.std(ddof=1)),"sem":float(a.std(ddof=1)/math.sqrt(N))}
        out["psnr_from_global_mse_db"]=float(-10*np.log10(out["mse_01"]["mean"]))
        return out

    results=[]
    previews={}
    all_features={}
    for label,path,ck in checkpoints:
        model=VAE(**ck["model_kwargs"]).to(DEVICE).eval()
        model.load_state_dict(ck["model"],strict=True)
        cached=cache_index.get(str(path.resolve()))
        if cached is not None:
            if cached["config"]!=ck["config"] or cached["model_kwargs"]!=ck["model_kwargs"] or cached["checkpoint_epoch"]!=args.epoch:
                raise ValueError(f"Cached checkpoint settings differ: {path}")
            count=min(BATCH,N)
            model_previews={}
            errors={}
            with torch.inference_mode():
                x=images[:count].to(DEVICE)
                mu,lv=model.encode(x)
                noise=torch.stack([
                    torch.randn(mu.shape[1:],generator=torch.Generator().manual_seed(SEED+image_id))
                    for image_id in image_ids[:count]
                ]).to(DEVICE)
                for mode in ["sample","mean"]:
                    latent=mu+(0.5*lv).exp()*noise if mode=="sample" else mu
                    reconstruction,_=model.decode(latent)
                    scores,prediction=paired_image_metrics(x,reconstruction)
                    for key in ["mse_01","mae_01","psnr_db","ssim","lpips_alex_v01","mse_raw_m11"]:
                        actual=scores[key].cpu().double().numpy()
                        expected=np.asarray(cached["per_image"][mode][key][:count])
                        if not np.allclose(actual,expected,rtol=5e-5,atol=1e-6):
                            raise ValueError(f"Cached metric check failed: {label}/{mode}/{key}")
                        errors[f"{mode}/{key}"]=float(np.max(np.abs(actual-expected)))
                    model_previews[mode]=list(prediction[:6].cpu().unbind(0))
            row=json.loads(json.dumps(cached))
            row["label"]=label
            row["evaluation_origin"]={"kind":"cached","source":str(Path(args.base_results).resolve()),
                                      "device":cached_payload["protocol"]["device"]}
            cache_checks[label]={"images_rechecked":count,"max_absolute_errors":errors}
            results.append(row)
            previews[label]=model_previews
            print("[cache verified]",label,"images",count,flush=True)
            del model
            continue
        C=ck["model_kwargs"]["z_channels"]
        zlv=float(ck["model"]["_encoder.z_logvar"])
        xlv=float(ck["model"]["_decoder.x_logvar"])
        zvar=math.exp(zlv);xvar=math.exp(xlv)
        beta=ck["config"]["model"]["beta"]
        values={mode:{} for mode in ["sample","mean"]}
        out_features={mode:[] for mode in values}
        post_kl=[]
        coordinate_sum=torch.zeros(C,32,32,dtype=torch.float64)
        coordinate_sq=torch.zeros_like(coordinate_sum)
        raw_moments=np.zeros(4,dtype=np.float64)
        channel_sum=torch.zeros(C,dtype=torch.float64)
        channel_outer=torch.zeros(C,C,dtype=torch.float64)
        model_previews={mode:[] for mode in values}
        print("[model]",label,flush=True)
        started=time.monotonic()
        with torch.inference_mode():
            for start in range(0,N,BATCH):
                x=images[start:start+BATCH].to(DEVICE)
                mu,lv=model.encode(x)
                assert lv.numel()==1 and tuple(mu.shape[1:])==(C,32,32)
                cpu_mu=mu.cpu().double()
                coordinate_sum+=cpu_mu.sum(0)
                coordinate_sq+=cpu_mu.square().sum(0)
                for k in range(1,5):raw_moments[k-1]+=float(cpu_mu.pow(k).sum())
                tokens=cpu_mu.permute(0,2,3,1).reshape(-1,C)
                channel_sum+=tokens.sum(0)
                channel_outer+=tokens.T@tokens
                kl=(0.5*(cpu_mu.square()+zvar-zlv-1).flatten(1).sum(1)).numpy()
                post_kl.extend(kl.tolist())
                noise=torch.stack([
                    torch.randn(mu.shape[1:],generator=torch.Generator().manual_seed(SEED+image_id))
                    for image_id in image_ids[start:start+len(x)]
                ]).to(DEVICE)
                for mode in values:
                    latent=mu+(0.5*lv).exp()*noise if mode=="sample" else mu
                    reconstruction,_=model.decode(latent)
                    scores,prediction=paired_image_metrics(x,reconstruction)
                    for key,value in scores.items():
                        values[mode].setdefault(key,[]).extend(value.cpu().double().tolist())
                    out_features[mode].append(features(prediction).cpu())
                    if start<6:model_previews[mode].extend(prediction[:6-start].cpu().unbind(0))
                if (start+len(x))%256==0 or start+len(x)==N:
                    print("[eval]",label,start+len(x),"/",N,"seconds",round(time.monotonic()-started,1),flush=True)

        dims=C*32*32
        coord_mean=coordinate_sum/N
        coord_mu_var=(coordinate_sq/N-coord_mean.square()).clamp_min(0)
        aggregate_coord_var=coord_mu_var+zvar
        mean,second,third,fourth=raw_moments/(N*dims)
        variance_mu=max(0.,second-mean**2)
        total_var=variance_mu+zvar
        central3=third-3*mean*second+2*mean**3
        central4=fourth-4*mean*third+6*mean**2*second-3*mean**4
        excess=(central4+6*zvar*variance_mu+3*zvar**2)/(total_var**2)-3
        token_n=N*32*32
        cmean=channel_sum/token_n
        cov=channel_outer/token_n-cmean[:,None]*cmean[None,:]+torch.eye(C,dtype=torch.float64)*zvar
        cstd=cov.diag().sqrt()
        corr=cov/(cstd[:,None]*cstd[None,:])
        off=corr[~torch.eye(C,dtype=torch.bool)].abs()
        latent_diagnostics={
            "conditional_kl_nats_per_image":float(np.mean(post_kl)),
            "conditional_kl_nats_per_latent":float(np.mean(post_kl)/dims),
            "posterior_z_logvar":zlv,"posterior_z_variance":zvar,
            "aggregate_pooled_mean":float(mean),"aggregate_pooled_std":math.sqrt(total_var),
            "aggregate_pooled_skewness":float(central3/total_var**1.5),"aggregate_pooled_excess_kurtosis":float(excess),
            "aggregate_diagonal_gaussian_moment_kl_nats_per_latent":float((.5*(coord_mean.square()+aggregate_coord_var-aggregate_coord_var.log()-1)).mean()),
            "channel_correlation_mean_abs":float(off.mean()),"channel_correlation_max_abs":float(off.max()),
            "active_coordinate_fraction_var_mu_gt_0.01":float((coord_mu_var>.01).double().mean()),
            "x_logvar":xlv,"x_variance":xvar,"beta_times_x_variance":beta*xvar,
        }
        row={"label":label,"run":str(path.parent.parent),"checkpoint":str(path),"checkpoint_epoch":args.epoch,
             "beta":beta,"z_channels":C,"latent_shape":[C,32,32],"latent_elements":dims,
             "element_compression_ratio":196608/dims,"parameters":sum(p.numel() for p in model.parameters()),
             "config":ck["config"],"model_kwargs":ck["model_kwargs"],
             "original_history":ck["history"],"latent":latent_diagnostics,"modes":{},"per_image":{},
             "per_image_conditional_kl_nats":post_kl,
             "evaluation_origin":{"kind":"new","device":str(DEVICE)}}
        for mode in values:
            f=torch.cat(out_features[mode])
            score=summarize(values[mode])
            score["rfid_2048"]=rfid(f)
            raw=np.array(values[mode]["mse_raw_m11"])
            nll=.5*(196608*raw/xvar+196608*(xlv+math.log(2*math.pi)))
            if mode=="sample":
                score["negative_elbo_nats_per_image_mc"]=float(np.mean(nll+np.array(post_kl)))
            score["mse_over_x_variance"]=float(raw.mean()/xvar)
            row["modes"][mode]=score
            row["per_image"][mode]=values[mode]
            print("[scores]",label,mode,json.dumps(score),flush=True)
        results.append(row)
        previews[label]=model_previews
        del model,out_features,values
        torch.cuda.empty_cache()

    comparisons={}
    pair_specs=[("z_channels_32_to_8",0,1),("beta_0.5_to_0.01",1,2)]
    beta_indices=sorted((i for i,r in enumerate(results) if r["z_channels"]==8),
                        key=lambda i:results[i]["beta"],reverse=True)
    existing_pairs={(i,j) for _,i,j in pair_specs}
    for i,j in zip(beta_indices,beta_indices[1:]):
        if (i,j) not in existing_pairs:
            pair_specs.append((f"beta_{results[i]['beta']:g}_to_{results[j]['beta']:g}",i,j))
    for effect,i,j in pair_specs:
        pair={"from":LABELS[i],"to":LABELS[j],"modes":{}}
        for mode in ["sample","mean"]:
            changes={}
            for key in ["mse_01","psnr_db","ssim","lpips_alex_v01","mse_raw_m11"]:
                a=np.asarray(results[i]["per_image"][mode][key])
                b=np.asarray(results[j]["per_image"][mode][key])
                delta=b-a;sem=float(delta.std(ddof=1)/math.sqrt(N))
                changes[key]={"delta_to_minus_from":float(delta.mean()),"ci95":[float(delta.mean()-1.96*sem),float(delta.mean()+1.96*sem)],
                              "relative_change_percent":float(100*(b.mean()/a.mean()-1))}
            changes["rfid_delta"]=results[j]["modes"][mode]["rfid_2048"]-results[i]["modes"][mode]["rfid_2048"]
            pair["modes"][mode]=changes
        comparisons[effect]=pair

    def html_table(headers,rows):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+str(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    def avg_sd(summary,key,digits):
        v=summary[key]
        return f"{v['mean']:.{digits}f} ± {v['std_across_images']:.{digits}f}"
    def main_table(mode):
        return html_table(["实验","z 通道","β","rFID ↓","PSNR (dB) ↑","SSIM ↑","LPIPS-Alex ↓","MSE [0,1] ↓"],[
            [r["label"].split(":")[0],r["z_channels"],r["beta"],f"{r['modes'][mode]['rfid_2048']:.4f}",
             avg_sd(r["modes"][mode],"psnr_db",3),avg_sd(r["modes"][mode],"ssim",4),avg_sd(r["modes"][mode],"lpips_alex_v01",4),
             f"{r['modes'][mode]['mse_01']['mean']:.7f}"] for r in results])
    def change_sentence(effect,mode):
        p=comparisons[effect]["modes"][mode]
        return f"PSNR 变化 {p['psnr_db']['delta_to_minus_from']:+.3f} dB，SSIM 变化 {p['ssim']['delta_to_minus_from']:+.4f}，LPIPS 变化 {p['lpips_alex_v01']['delta_to_minus_from']:+.4f}，rFID 变化 {p['rfid_delta']:+.4f}；MSE 相对变化 {p['mse_01']['relative_change_percent']:+.2f}%"
    def png_data(grid):
        b=io.BytesIO();grid.save(b,format="PNG")
        return "data:image/png;base64,"+base64.b64encode(b.getvalue()).decode("ascii")
    def image_grid(mode):
        # Figure pixels only; captions are HTML so no font-cache files are needed.
        h,w=256,256
        columns=[images[:6].add(1).mul(.5).clamp(0,1),*[torch.stack(previews[label][mode]) for label in LABELS]]
        canvas=Image.new("RGB",(len(columns)*w,min(6,N)*h),"white")
        for col,batch in enumerate(columns):
            for row,img in enumerate(batch):
                array=(img.permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8)
                canvas.paste(Image.fromarray(array),(col*w,row*h))
        return png_data(canvas)

    import html
    settings_table=html_table(["实验","训练目录","Epoch","潜空间 C×H×W","潜元素数","输入/潜元素数","AE 参数 (M)"],[
        [r["label"],Path(r["run"]).name,args.epoch,"×".join(map(str,r["latent_shape"])),r["latent_elements"],f"{r['element_compression_ratio']:.0f}×",f"{r['parameters']/1e6:.4f}"] for r in results])
    latent_table=html_table(["实验","KL nats/image","KL nats/latent","后验方差 exp(z_logvar)","聚合 mean / std","逐坐标高斯 KL 代理 / latent","通道 |corr| 均值 / 最大值"],[
        [r["label"].split(":")[0],f"{r['latent']['conditional_kl_nats_per_image']:.2f}",f"{r['latent']['conditional_kl_nats_per_latent']:.5f}",
         f"{r['latent']['posterior_z_variance']:.6f}",f"{r['latent']['aggregate_pooled_mean']:.5f} / {r['latent']['aggregate_pooled_std']:.5f}",
         f"{r['latent']['aggregate_diagonal_gaussian_moment_kl_nats_per_latent']:.5f}",
         f"{r['latent']['channel_correlation_mean_abs']:.5f} / {r['latent']['channel_correlation_max_abs']:.5f}"] for r in results])
    variance_table=html_table(["实验","exp(x_logvar)","β exp(x_logvar)","采样 raw MSE","均值 raw MSE","采样 MSE / x_var","测试负 ELBO nats/image"],[
        [r["label"].split(":")[0],f"{r['latent']['x_variance']:.6f}",f"{r['latent']['beta_times_x_variance']:.7f}",
         f"{r['modes']['sample']['mse_raw_m11']['mean']:.6f}",f"{r['modes']['mean']['mse_raw_m11']['mean']:.6f}",
         f"{r['modes']['sample']['mse_over_x_variance']:.4f}",f"{r['modes']['sample']['negative_elbo_nats_per_image_mc']:.1f}"] for r in results])
    delta_rows=[]
    for effect,pair in comparisons.items():
        for mode,changes in pair["modes"].items():
            for key in ["psnr_db","ssim","lpips_alex_v01","mse_01"]:
                d=changes[key];lo,hi=d["ci95"]
                delta_rows.append([effect,mode,key,f"{d['delta_to_minus_from']:+.7f}",f"[{lo:+.7f}, {hi:+.7f}]"])
    delta_table=html_table(["比较（后减前）","模式","指标","平均配对差","近似 95% 配对区间"],delta_rows)

    protocol={"dataset":"MM-CelebA-HQ","total_images":len(dataset),"train_images":n_train,"val_images":n_val,"test_images":N,
              "split_seed":base_config["seed"],"test_indices":test_indices,"test_image_ids":image_ids,"posterior_seed_base":SEED,
              "posterior_noise":"per image: CPU Generator.manual_seed(seed_base + numeric image_id); first 8 channels shared for C8/C32",
              "posterior_samples_per_image":1,"primary_mode":"sample","secondary_mode":"mean",
              "input_resize":[256,256],"preprocessing":"PIL RGB -> torchvision v2 uint8 bilinear resize antialias=True -> float32 [0,1] -> Normalize(.5,.5)",
              "image_metrics":"RGB, recon clipped to [0,1]; MSE/MAE average all C,H,W; per-image PSNR then average",
              "ssim":"RGB channel mean, valid Gaussian 11x11 sigma1.5, population moments, constants .01^2/.03^2, data_range=1",
              "lpips":"Official PerceptualSimilarity code, AlexNet, v0.1, learned calibration, ImageNet trunk, eval mode; clipped images mapped to [-1,1]",
              "rfid":"Official pytorch-fid TensorFlow-port Inception pool3 2048, bilinear resize299 align_corners=False, covariance denominator N-1, continuous float32 images",
              "rfid_solver":"FP64 symmetric PSD square roots via eigh/eigvalsh on GPU; negative roundoff eigenvalues clipped to zero",
              "precision":"FP32 eager VAE/LPIPS/Inception, FP64 FID and aggregate summaries; no AMP, no TF32",
              "device":str(DEVICE),"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"torch":torch.__version__,"torchvision":torchvision.__version__,"batch_size":BATCH}
    payload={"created_at":datetime.now().astimezone().isoformat(),"protocol":protocol,"downloads":download_manifest,"experiments":results,"paired_comparisons":comparisons,
             "cache_reuse":{"base_results":str(Path(args.base_results).resolve()) if args.base_results else None,
                            "base_results_sha256":hashlib.sha256(Path(args.base_results).read_bytes()).hexdigest() if args.base_results else None,
                            "checks":cache_checks}}
    raw_json=json.dumps(payload,ensure_ascii=False,allow_nan=False)
    csv_io=io.StringIO()
    writer=csv.writer(csv_io)
    writer.writerow(["experiment","z_channels","beta","mode","N","rfid_2048","psnr_mean_db","psnr_sd","ssim_mean","ssim_sd","lpips_alex_v01_mean","lpips_sd","mse_01","kl_nats_image","kl_nats_latent"])
    for r in results:
        for mode,s in r["modes"].items():
            writer.writerow([r["label"],r["z_channels"],r["beta"],mode,N,s["rfid_2048"],s["psnr_db"]["mean"],s["psnr_db"]["std_across_images"],s["ssim"]["mean"],s["ssim"]["std_across_images"],s["lpips_alex_v01"]["mean"],s["lpips_alex_v01"]["std_across_images"],s["mse_01"]["mean"],r["latent"]["conditional_kl_nats_per_image"],r["latent"]["conditional_kl_nats_per_latent"]])
    def data_uri(content,mime):
        return "data:"+mime+";base64,"+base64.b64encode(content.encode()).decode()
    latex_lines=[r"\begin{tabular}{lrrrrrr}",r"\toprule",r"Model & $c$ & $\beta$ & rFID$\downarrow$ & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ \\",r"\midrule"]
    for r in results:
        s=r["modes"]["sample"]
        latex_lines.append(f"{r['label'].split(':')[0]} & {r['z_channels']} & {r['beta']} & {s['rfid_2048']:.4f} & {s['psnr_db']['mean']:.3f} & {s['ssim']['mean']:.4f} & {s['lpips_alex_v01']['mean']:.4f} "+r"\\")
    latex_lines += [r"\bottomrule",r"\end{tabular}"]
    latex="\n".join(latex_lines)

    # Data-dependent interpretation; do not prescribe an outcome before measuring.
    a,b,c=results[:3]
    z_score=comparisons["z_channels_32_to_8"]["modes"]["sample"]
    beta_score=comparisons["beta_0.5_to_0.01"]["modes"]["sample"]
    z_better=sum([z_score["psnr_db"]["delta_to_minus_from"]>0,z_score["ssim"]["delta_to_minus_from"]>0,z_score["lpips_alex_v01"]["delta_to_minus_from"]<0,z_score["rfid_delta"]<0])
    beta_better=sum([beta_score["psnr_db"]["delta_to_minus_from"]>0,beta_score["ssim"]["delta_to_minus_from"]>0,beta_score["lpips_alex_v01"]["delta_to_minus_from"]<0,beta_score["rfid_delta"]<0])
    z_verdict=("本次 C8 在这四项指标上均优于 C32。" if z_better==4 else "本次 C32 在这四项指标上均优于 C8。" if z_better==0 else "本次通道数比较存在指标间的折中，不能只按一个分数选模型。")
    beta_verdict=("降低 β 后，这四项重建指标均改善。" if beta_better==4 else "降低 β 后，这四项重建指标均未改善。" if beta_better==0 else "降低 β 后，重建指标存在改善与退步并存的折中。")
    noise_ratio=c["latent"]["posterior_z_variance"]/b["latent"]["posterior_z_variance"]
    kl_ratio=c["latent"]["conditional_kl_nats_per_image"]/b["latent"]["conditional_kl_nats_per_image"]
    mode_disagreement=[]
    for effect in comparisons:
        for key in ["psnr_db","ssim","lpips_alex_v01"]:
            x=comparisons[effect]["modes"]["sample"][key]["delta_to_minus_from"]
            y=comparisons[effect]["modes"]["mean"][key]["delta_to_minus_from"]
            if x*y<0:mode_disagreement.append(effect+"/"+key)
    mode_note=("采样/均值模式存在排序反转："+", ".join(mode_disagreement)+"。因此容量与 β 的结论必须注明后验使用方式。" if mode_disagreement else "本次 PSNR、SSIM、LPIPS 在采样与均值模式下的单因素变化方向一致；幅度仍会随模式改变。")
    best={key:min(results,key=lambda r:r["modes"]["sample"][key])["label"] for key in ["rfid_2048"]}
    best.update({key:(max if high else min)(results,key=lambda r:r["modes"]["sample"][key]["mean"])["label"] for key,high in [("psnr_db",True),("ssim",True),("lpips_alex_v01",False)]})
    best_line="；".join(f"{key}: {label}" for key,label in best.items())
    capacity_delta=a["parameters"]-b["parameters"]
    report=f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>{len(results)}组 VAE 实验对比 · {datetime.now():%Y-%m-%d}</title><style>
    body{{font:16px/1.7 system-ui,sans-serif;max-width:1400px;margin:auto;padding:28px;color:#182739;background:#fafbfd}}
    h1,h2{{line-height:1.35}}h2{{margin-top:2em}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}
    th,td{{padding:10px;border-bottom:1px solid #dce3eb;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}
    .scroll{{overflow-x:auto;margin:16px 0}}.card{{background:white;padding:18px 24px;border:1px solid #dce3eb;border-radius:10px}}
    pre{{white-space:pre-wrap;overflow-wrap:anywhere;padding:16px;background:#edf2f7}}img{{width:100%;height:auto}}a{{color:#2269a1}}.muted{{color:#596a7b}}
    </style><h1>{len(results)}组 VAE 实验的统一评估与单因素比较</h1>
    <p class="muted">{payload['created_at']} · MM-CelebA-HQ 256×256 · 固定测试集评估 N={N} · 全部 epoch {args.epoch}</p>
    <div class="card"><p><strong>主表使用一次固定后验采样，另附确定性的均值重建表；所有实验使用相同图片、指标实现和数值精度。</strong></p>
    <p>{z_verdict}{beta_verdict}</p><p>主表各项最佳：{html.escape(best_line)}。</p>
    <p>本报告包含：{'；'.join(html.escape(r['label']) for r in results)}。设置来自实际 checkpoint；配置文件本身不算已完成实验。</p>
    <p>复用结果 {'、'.join(html.escape(r['label'].split(':')[0]) for r in results if r['evaluation_origin']['kind']=='cached') or '无'}；
    本次完整补算 {'、'.join(html.escape(r['label'].split(':')[0]) for r in results if r['evaluation_origin']['kind']=='new') or '无'}。
    缓存复用时核对完整测试图片顺序、库版本、指标源码/权重 SHA256，并重新评估每个缓存模型的首批图片；仍假设原始数据集内容保持不变。</p></div>

    <h2>1. 实验设置与容量</h2>{settings_table}
    <p>全部使用 seed=42、24000/3000/3000 划分、相同学习率/调度、batch size 16×2 卡、{args.epoch} epochs、init_channels=64、ch_mult=[1,2,4,4]。
    空间下采样均为 f=8（256→32），空间 token 数均为 1024；只有 C32→C8 改变潜通道数。
    输入/潜元素数是标量元素压缩倍率，不是熵编码后的码率或 bits/pixel。</p>

    <h2>2. 论文式主表：后验采样重建</h2>{main_table('sample')}
    <p>重建方式：z=mu(x)+exp(z_logvar/2)·epsilon。每图一次采样，seed={SEED}+图片数字 ID；所有 C8 实验噪声完全相同，C32 实验与其前 8 个噪声通道相同。
    ± 为测试图片之间的样本标准差，不是多次训练的均值±标准差，也不是置信区间。
    rFID 是真实测试图片与其重建图片之间的 2048 维 Inception 特征 Fréchet 距离；不是从先验随机生成图片的 FID。</p>

    <h2>3. 补充表：后验均值重建</h2>{main_table('mean')}
    <p>重建方式：z=mu(x)。这张表衡量确定性重建，不是训练期望重建项的 Monte Carlo 估计。{html.escape(mode_note)}
    PSNR 使用逐图 PSNR 后平均；不能与“先对全数据平均 MSE 再转 PSNR”混用。后者已保存在完整 JSON 中。</p>

    <h2>4. 潜空间与正则化诊断</h2>{latent_table}
    <p>KL 是含 1/2 的标准 E_x KL(q(z|x)||N(0,I))。它不等于 KL(q(z)||N(0,I))；按 latent 元素归一化后更适合比较不同 C 的正则化强度。
    “逐坐标高斯 KL 代理”只用经验聚合后验的均值/方差做高斯矩匹配，忽略非高斯性与坐标相关，不能当作真实聚合 KL。
    聚合统计使用 Var_q(z)=Var_x(mu)+E_x[var(z|x)]，通道相关将图片与空间位置合并为 token。</p>
    {variance_table}
    <p>raw MSE 在未裁剪的 [-1,1] 尺度上计算；主表 MSE 使用裁剪后的 [0,1] 图像，因此不能简单把两者统一除以 4（越界像素比例已保存）。
    负 ELBO 使用当前高斯观测模型、包含 log(2π) 和 1/2、β=1 的标准 KL，在采样模式下估计，仅作为模型目标诊断；
    这是连续密度的负 ELBO（nats/image），不是离散 8-bit 图像 NLL/BPD，也不能替代感知质量指标。</p>

    <h2>5. z_channels 的影响：A → B</h2>
    <p><strong>控制 β=0.5，将 C 从 32 降到 8，潜元素数减少 75%，同精度潜张量存储降到 1/4，元素压缩倍率由 6×提高到 24×。</strong>
    AE 参数只减少 {capacity_delta:,}（{100*capacity_delta/a['parameters']:.3f}%），因为主体卷积/注意力仍在相同隐藏通道数上计算；
    不能由潜通道降到 1/4 推断整个 VAE 或后续扩散模型的 FLOPs 也降到 1/4。</p>
    <p>采样模式：{change_sentence('z_channels_32_to_8','sample')}。{z_verdict}</p>
    <p>均值模式：{change_sentence('z_channels_32_to_8','mean')}。</p>
    <p>容量增加提供更大的表示空间，但不会保证固定训练预算下的重建一定更好。
    当前 KL 对所有潜元素求和，通道数改变也改变了正则化项的总量；共享 z_logvar 又允许不同容量选择不同的注入噪声。
    因此这组对照反映“当前训练目标下改变 C 的综合影响”，不是剥离正则化和采样噪声后的纯容量上限。
    更严谨的后续消融应匹配 KL 预算/率失真工作点并增加训练随机种子。</p>

    <h2>6. beta 的影响：B → C</h2>
    <p><strong>控制 C=8，将 β 从 0.5 降为 0.01，显式 KL 系数减小 50 倍。</strong>
    采样模式：{change_sentence('beta_0.5_to_0.01','sample')}。{beta_verdict}</p>
    <p>均值模式：{change_sentence('beta_0.5_to_0.01','mean')}。</p>
    <p>平均条件 KL 变为 B 的 {kl_ratio:.3f} 倍，后验采样噪声方差变为 B 的 {noise_ratio:.3f} 倍。
    这两项反映逐样本后验约束与注入噪声的变化；条件 KL 同时包含互信息和聚合先验失配，不能直接当作有效信息量；
    不能把重建改善全部归因为“噪声变小”，因为编码器和解码器参数也一起重新优化了。</p>
    <p>较小 β 对逐样本后验的先验约束更弱。是否损害聚合先验匹配，需结合上述聚合统计、相关性与高斯代理观察；
    平均条件 KL 上升本身不能直接证明真实聚合 KL 一定上升。
    这些实验没有测量先验随机采样生成 FID 或训练后续扩散模型，因此不能据此断言生成质量、可解耦性或下游 LDM 质量提高。</p>
    <p>当前目标为 L_code=D·(MSE/exp(s)+s)+β·K_code，其中 K_code=2·KL_standard。
    对固定 s，网络梯度中的相对 KL/平方误差系数可写作 β·exp(s)，所以表中同时列出了这个值。
    它只解释固定方差下的相对梯度权重，不代表整个 Adam 动态的严格等价。
    原始训练 total loss 同时依赖 β 和可学习 s，不作为跨实验的优劣排名。</p>

    {interpretation_appendix(payload)}
    {beta_sweep_appendix(payload)}
    <h2>7. 配对变化与结论边界</h2>{delta_table}
    <p>区间为配对逐图差的 mean±1.96·SE，反映本测试集和固定采样噪声下的变化；
    不是独立重复训练的不确定性，不覆盖不同初始化/数据顺序，不能当作超参数结论的多 seed 显著性证据。
    rFID 是整组统计量，不把它伪造成逐图分数或附上逐图标准差。本次 N={N}，仍有有限样本偏差；
    这里只支持相同数据、相同特征和预处理下的内部比较，不直接与其他论文不同数据集或样本量的数值横向排名。</p>

    <h2>8. 同一组图片的重建对照</h2><p>列顺序：输入｜{'｜'.join(html.escape(r['label']) for r in results)}。使用测试划分中的前 {min(6,N)} 张，未按指标挑图。</p>
    <h3>后验采样</h3><img id="sample-grid" alt="同图后验采样重建对比" src="{image_grid('sample')}">
    <h3>后验均值</h3><img id="mean-grid" alt="同图后验均值重建对比" src="{image_grid('mean')}">

    <h2>9. 可用于论文的描述与 LaTeX</h2>
    <p>“我们在 MM-CelebA-HQ 的固定 {N} 张测试图片上比较 {len(results)} 个训练 {args.epoch} epochs 的 VAE，并在相同 f=8 下考察潜通道数及 KL 权重。
    我们报告重建 rFID、逐图平均 PSNR、SSIM 和 LPIPS-Alex v0.1，同时提供后验均值与固定噪声采样两种重建协议。
    在采样协议下，C32→C8（β=0.5）的结果为：{change_sentence('z_channels_32_to_8','sample')}；
    β=0.5→0.01（C8）的结果为：{change_sentence('beta_0.5_to_0.01','sample')}。
    {' '.join('新增 '+html.escape(effect)+'：'+change_sentence(effect,'sample')+'。' for effect in comparisons if effect not in ['z_channels_32_to_8','beta_0.5_to_0.01'])}
    这些结果来自单次训练，反映当前架构和预算下的率失真折中。”</p>
    <pre>{html.escape(latex)}</pre><p>LaTeX 表使用 booktabs；正式稿的 caption 需注明数据集、N={N}、epoch {args.epoch}、采样协议及 LPIPS 版本。</p>

    <h2>10. 数据下载、可复现口径与参考</h2>
    <p><a download="vae_comparison.csv" href="{data_uri(csv_io.getvalue(),'text/csv')}">下载汇总 CSV</a> ·
    <a download="vae_comparison_full.json" href="{data_uri(raw_json,'application/json')}">下载完整结果与逐图数据 JSON</a> ·
    <a download="vae_comparison.tex" href="{data_uri(latex,'text/plain')}">下载 LaTeX 表</a></p>
    <p>本报告嵌入全部图像与可下载数据，同目录同时导出 CSV、JSON、LaTeX 和 PNG。预训练权重与上游指标源码只载入内存，不安装依赖或写权重缓存。
    JSON 包含每个 checkpoint 的实际配置、历史 loss、测试图片 ID/划分索引、逐图指标、采样种子、来源 URL 与 SHA256。</p>
    <pre>{html.escape(json.dumps({k:v for k,v in protocol.items() if k not in ['test_indices','test_image_ids']},ensure_ascii=False,indent=2))}</pre>
    <p>指标选择参考 <a href="https://arxiv.org/html/2112.10752v2">LDM（Table 8，R-FID/PSNR/SSIM 等）</a>；
    <a href="https://par.nsf.gov/servlets/purl/10647601">DC-AE（rFID/PSNR/SSIM/LPIPS 对比表）</a>；
    LPIPS 使用 <a href="https://github.com/richzhang/PerceptualSimilarity">作者官方实现</a>；
    rFID 使用 <a href="https://github.com/mseitzer/pytorch-fid">pytorch-fid 的 FID 专用 Inception 与权重</a>。</p>
    <script id="comparison-data" type="application/json">{raw_json.replace('<',chr(92)+'u003c')}</script></html>"""

    with OUTPUT.open("x",encoding="utf-8") as handle:handle.write(report)

    # Export the same data embedded in HTML as ordinary result files.
    (OUTPUT_DIR/"metrics.json").write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False,indent=2)+"\n",encoding="utf-8")
    (OUTPUT_DIR/"summary.csv").write_text(csv_io.getvalue(),encoding="utf-8")
    (OUTPUT_DIR/"comparison_table.tex").write_text(latex+"\n",encoding="utf-8")
    with (OUTPUT_DIR/"per_image_metrics.csv").open("x",newline="",encoding="utf-8") as handle:
        fields=["experiment","mode","image_id","conditional_kl_nats","mse_01","mae_01","psnr_db","ssim","lpips_alex_v01","mse_raw_m11","clipped_pixel_fraction"]
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for row in results:
            for mode,values in row["per_image"].items():
                for index,image_id in enumerate(image_ids):
                    writer.writerow({"experiment":row["label"],"mode":mode,"image_id":image_id,
                                     "conditional_kl_nats":row["per_image_conditional_kl_nats"][index],
                                     **{key:values[key][index] for key in fields[4:]}})
    for mode in ["sample","mean"]:
        encoded=re.search(r'id="'+mode+r'-grid"[^>]*src="data:image/png;base64,([^"]+)"',report).group(1)
        (OUTPUT_DIR/f"reconstructions_{mode}.png").write_bytes(base64.b64decode(encoded))

    print("[REPORT]",OUTPUT,"bytes",OUTPUT.stat().st_size,flush=True)
    print("[FINAL_SUMMARY]",json.dumps({"best":best,"comparisons":comparisons,"summary":[{"label":r["label"],"sample":r["modes"]["sample"],"mean":r["modes"]["mean"],"latent":r["latent"]} for r in results]},ensure_ascii=False,allow_nan=False),flush=True)


if __name__ == "__main__":
    main()
