import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from models.backbone import Backbone

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Disentangler(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.fc = nn.Linear(emb_dim, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)
    def forward(self, x):
        x = F.relu(self.bn(self.fc(x)))
        x = F.dropout(x, training=self.training)
        return x

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=[]):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class Transformer(nn.Module):
    def __init__(self, emb_dim, attr_dim, hidden=[]):
        super().__init__()
        self.att = MLP(attr_dim, emb_dim, hidden)
        self.fc = MLP(emb_dim + attr_dim, emb_dim, hidden)
    def forward(self, img_feat, attr_feat):
        att = torch.sigmoid(self.att(attr_feat))
        img_feat = img_feat * att + img_feat
        combined = torch.cat([img_feat, attr_feat], dim=1)
        return self.fc(combined)

class SymNet(nn.Module):
    def __init__(self, dset, args):
        super().__init__()
        self.dset = dset
        self.emb_dim = args.emb_dim
        self.drop = args.drop
        self.lambda_rep = args.lambda_rep
        self.lambda_grad = args.lambda_grad

        self.feat_extractor = Backbone('resnet18') # e.g., resnet18
        feat_dim = 512
        self.img_embedder = nn.Sequential(
            nn.Conv2d(feat_dim, args.emb_dim, 1, bias=False),
            nn.BatchNorm2d(args.emb_dim),
            nn.ReLU()
        )
        self.img_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.D = nn.ModuleDict({
            'da': Disentangler(args.emb_dim),
            'do': Disentangler(args.emb_dim)
        })

        self.transformer_attr = Transformer(args.emb_dim, args.emb_dim)
        self.transformer_obj = Transformer(args.emb_dim, args.emb_dim)

        self.attr_clf = MLP(args.emb_dim, len(dset.attrs))
        self.obj_clf = MLP(args.emb_dim, len(dset.objs))

    def RMD_prob(self, feat_plus, feat_minus, img_feat):
        d_plus = torch.norm(feat_plus - img_feat, dim=1)
        d_minus = torch.norm(feat_minus - img_feat, dim=1)
        return F.softmax(d_minus - d_plus, dim=0)

    def train_forward(self, x,epoch):
        img, attrs, objs, pos_attr_img, pos_obj_img = x[:5]
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]
        pos_attr_img, neg_objs, pos_obj_img, neg_attrs = x[4], x[5], x[6], x[7]
        neg_obj_pairs, neg_attr_pairs = x[10], x[11]
        img = self.feat_extractor(img)[0]
        img = self.img_embedder(img)
        img_feat = self.img_avg_pool(img).squeeze()
        pos_attr_img = self.feat_extractor(pos_attr_img)[0]
        pos_attr_img = self.img_embedder(pos_attr_img)
        pos_attr_feat = self.img_avg_pool(pos_attr_img).squeeze()
        pos_obj_img = self.feat_extractor(pos_obj_img)[0]
        pos_obj_img = self.img_embedder(pos_obj_img)
        pos_obj_feat = self.img_avg_pool(pos_obj_img).squeeze()
        feat_plus = self.transformer_attr(img_feat, pos_attr_feat)
        feat_minus = self.transformer_obj(img_feat, pos_obj_feat)

        attr_pred = self.attr_clf(self.D['da'](img_feat))
        obj_pred = self.obj_clf(self.D['do'](img_feat))
        attr_pos_pred = self.attr_clf(self.D['da'](pos_attr_feat))
        obj_pos_pred = self.obj_clf(self.D['do'](pos_obj_feat))

        loss_attr = F.cross_entropy(attr_pred, attrs)
        loss_obj = F.cross_entropy(obj_pred, objs)
        loss_pos_attr = F.cross_entropy(attr_pos_pred, attrs)
        loss_pos_obj = F.cross_entropy(obj_pos_pred, objs)

        rmd_prob = self.RMD_prob(feat_plus, feat_minus, img_feat)
	loss_rmd = F.mse_loss(rmd_prob, attrs.float())
        loss = 0.1 * loss_attr + loss_obj + 0.1 * loss_pos_attr + loss_pos_obj + loss_sym +loss_rmd
        return loss, {'attr': attr_pred, 'obj': obj_pred, 'rmd': rmd_prob}

    def val_forward(self, x,epoch):
        img, attrs, objs, _ = x[:4]
        img = x[0]
        img = self.feat_extractor(img)[0]
        img = self.img_embedder(img)
        img_feat = self.img_avg_pool(img).squeeze()
        attr_pred = F.softmax(self.attr_clf(self.D['da'](img_feat)), dim=1)
        obj_pred = F.softmax(self.obj_clf(self.D['do'](img_feat)), dim=1)
        scores = {}
        for attr, obj in self.dset.pairs:
            aid, oid = self.dset.attr2idx[attr], self.dset.obj2idx[obj]
            scores[(attr,obj)] = attr_pred[:, aid] * obj_pred[:, oid]
        return None, scores

    def forward(self, x,epoch):
        if self.training:
            return self.train_forward(x,epoch)
        else:
            with torch.no_grad():
                return self.val_forward(x,epoch)
