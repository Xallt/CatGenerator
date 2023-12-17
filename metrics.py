from piq import ssim, FID
from collections import defaultdict
import torch
from tqdm.auto import tqdm


def compute_metrics(test_loader, model, fid_num_samples=1000):
    fid_metric = FID()
    metrics = defaultdict(float)
    count = 0
    model.eval()
    for x in tqdm(test_loader):
        x = x.cuda()
        with torch.no_grad():
            x_enc = model.encoder(x)[0]
            x_rec = model.decoder(x_enc)
            metrics["ssim"] += ssim(x * 0.5 + 0.5, x_rec * 0.5 + 0.5).item()
        count += len(x)
    metrics = {k: v / count for k, v in metrics.items()}

    class FIDWrapperLoader:
        def __init__(self, dl):
            self.dl = dl

        def __iter__(self):
            for x in self.dl:
                yield {"images": (x * 0.5 + 0.5).clip(0, 1)}

    with torch.no_grad():
        model_samples = model.sample(fid_num_samples)
        model_samples_loader = torch.utils.data.DataLoader(model_samples, batch_size=64)

        test_features = fid_metric.compute_feats(FIDWrapperLoader(test_loader))
        model_features = fid_metric.compute_feats(
            FIDWrapperLoader(model_samples_loader)
        )
        metrics["fid"] = fid_metric(model_features, test_features).item()
    return metrics
