
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from simtext import similarity
from modeling_t5 import VLT5
class VLT5VCR(VLT5):
    def __init__(self, config):
        super().__init__(config)

        self.kl_div = nn.KLDivLoss(reduction='none')
        self.sim = similarity()

    def train_step(self, batch):
        self.train()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        vis_pos = batch['boxes'].to(device)

        B = vis_pos.size(0)
        V_L = vis_pos.size(1)

        log_train_accuracy = batch['log_train_accuracy']

        result = {}

        # QA
        qa_input_ids = batch['qa_input_ids'].to(device)
        qa_target_ids = batch['qa_target_ids'].to(device)

        vis_feats = vis_feats.unsqueeze(1).expand(-1, 4, -1, -1).contiguous().view(B*4, V_L, 2048)
        vis_pos = vis_pos.unsqueeze(1).expand(-1, 4, -1, -1).contiguous().view(B*4, V_L, 4)

        all_sentence = [0]*len(batch['qa_target_texts'])
        all_sentence_R = [0]*len(batch['qa_target_texts'])
        for i in range(len(batch['qa_target_texts'])):
            if batch['qa_target_texts'][i] == 'true':
                j = i // 4
                all_sentence[4 * j] = batch['qa_input_texts'][i]
                all_sentence[4 * j+1] = batch['qa_input_texts'][i]
                all_sentence[4 * j+2] = batch['qa_input_texts'][i]
                all_sentence[4 * j+3] = batch['qa_input_texts'][i]
        for i in range(len(batch['qar_target_texts'])):
            if batch['qar_target_texts'][i] == 'true':
                j = i // 4
                all_sentence_R[4 * j] = batch['qar_input_texts'][i]
                all_sentence_R[4 * j+1] = batch['qar_input_texts'][i]
                all_sentence_R[4 * j+2] = batch['qar_input_texts'][i]
                all_sentence_R[4 * j+3] = batch['qar_input_texts'][i]
        target_word = "answer:"
        target_word_R = "rationale:"
        before_sentence = [0]*len(batch['qa_target_texts'])
        after_sentence = [0]*len(batch['qa_target_texts'])
        after_sentence_R = [0] * len(batch['qa_target_texts'])
        True_sentence = [0] * len(batch['qa_target_texts'])
        for i in range(len(batch['qa_target_texts'])):
            target_index = all_sentence[i].index(target_word)
            before_sentence[i] = all_sentence[i][:target_index].strip()
            before_sentence[i] = before_sentence[i] + str(' Reasoning caption is')
            after_sentence[i] = all_sentence[i][target_index + len(target_word):].strip()
        for i in range(len(batch['qar_target_texts'])):
            target_index = all_sentence_R[i].index(target_word_R)
            after_sentence_R[i] = all_sentence_R[i][target_index + len(target_word_R):].strip()
            True_sentence[i] = 'true'

        before_ids = self.tokenizer(before_sentence, padding=True, truncation=True,max_length=30, return_tensors='pt').input_ids
        after_ids = self.tokenizer(after_sentence, padding=True, truncation=True,max_length=30, return_tensors='pt').input_ids
        after_ids_R = self.tokenizer(after_sentence_R, padding=True, truncation=True, max_length=35, return_tensors='pt').input_ids
        True_sentence = self.tokenizer(True_sentence, padding=True, truncation=True, max_length=35, return_tensors='pt').input_ids

        before_ids = before_ids.to(device)
        after_ids = after_ids.to(device)
        after_ids_R = after_ids_R.to(device)
        True_sentence = True_sentence.to(device)


        Q_Rt = torch.cat((before_ids,after_ids),1)
        A_Rt = torch.cat((before_ids,after_ids_R),1)


        # image and R similarity yuan
        before_ids_float = before_ids.float()
        i = after_ids_R.shape[0]
        k = i // 4
        xx = torch.zeros(k, k).to(device)
        for i in range(k):
            for j in range(k):
                xx[i, j] = torch.nn.functional.cosine_similarity(before_ids_float[4*i], before_ids_float[4*j],dim=0)
        yy = torch.eye(k, k).to(device)
        xx = xx - yy
        value, index = torch.topk(xx, k=1)
        value1, index1 = torch.topk(value, k=1, dim=0)
        i = index1[0][0].item()
        j = index[i][0].item()
        res = self.sim.compute(after_sentence_R[4*i],after_sentence_R[4*j])
        loss_Rsim = 1 - res['Sim_Cosine']

        objects_ids = batch['objects_ids']
        objects_ids_y = torch.zeros(len(objects_ids),10).to(device)
        for i in range(len(objects_ids)):
            kk = objects_ids[i].shape[1]
            if kk<=9:
                objects_ids_y[i][:kk] = objects_ids[i][:,:kk]
            else:
                objects_ids_y[i] = objects_ids[i][:,:10]


        output_caption = self(
            input_ids=before_ids,
            objects_ids=objects_ids_y,
            vis_inputs=(vis_feats, vis_pos),
            labels=after_ids_R,
            return_dict=True
        )

        qa_cap_loss = output_caption['loss'].mean()


        loss_wordsim = torch.nn.functional.cosine_similarity(output_caption['logits'][4 * i],
                                                             output_caption['logits'][4 * j])
        loss_sensim = loss_wordsim.mean()

        loss_R = abs(loss_Rsim - loss_sensim)

        output = self(
            input_ids=qa_input_ids,
            objects_ids=objects_ids_y,
            vis_inputs=(vis_feats, vis_pos),
            labels=qa_target_ids,
            return_dict=True
        )

        assert 'loss' in output

        qa_loss = output['loss'].mean()

        if log_train_accuracy:
            with torch.no_grad():
                logits = output['logits'].detach()
                logits = logits.view(B*4, self.lm_head.out_features)
                true_logit = logits[:, self.true_id].view(B*4, 1)
                false_logit = logits[:, self.false_id].view(B*4, 1)

                # [B*4, 2]
                binary_logits = torch.cat([true_logit, false_logit], dim=1)

                # [B*4]
                confidence = torch.softmax(binary_logits, dim=1)[:, 0]
                answer_id = confidence.view(B, 4).argmax(dim=1)

                result['qa_pred'] = answer_id.long().cpu().numpy()

        # QAR
        qar_input_ids = batch['qar_input_ids'].to(device)
        qar_target_ids = batch['qar_target_ids'].to(device)

        output = self(
            input_ids=qar_input_ids,
            objects_ids=objects_ids_y,
            vis_inputs=(vis_feats, vis_pos),
            labels=qar_target_ids,
            return_dict=True
        )
        assert 'loss' in output

        qar_loss = output['loss'].mean()

        if log_train_accuracy:
            with torch.no_grad():
                logits = output['logits'].detach()
                logits = logits.view(B*4, self.lm_head.out_features)
                true_logit = logits[:, self.true_id].view(B*4, 1)
                false_logit = logits[:, self.false_id].view(B*4, 1)

                # [B*4, 2]
                binary_logits = torch.cat([true_logit, false_logit], dim=1)

                # [B*4]
                confidence = torch.softmax(binary_logits, dim=1)[:, 0]
                rationale_id = confidence.view(B, 4).argmax(dim=1)

                result['qar_pred'] = rationale_id.long().cpu().numpy()

        loss = 1*qa_loss + 1*qar_loss + 3* qa_cap_loss

        result['loss'] = loss
        result['qa_loss'] = qa_loss.detach()
        result['qar_loss'] = qar_loss.detach()

        return result

    @torch.no_grad()
    def valid_step(self, batch):
        self.eval()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        vis_pos = batch['boxes'].to(device)

        B = vis_pos.size(0)
        V_L = vis_pos.size(1)

        result = {}

        vis_feats = vis_feats.unsqueeze(
            1).expand(-1, 4, -1, -1).contiguous().view(B*4, V_L, 2048)
        vis_pos = vis_pos.unsqueeze(
            1).expand(-1, 4, -1, -1).contiguous().view(B*4, V_L, 4)


        # QA
        qa_input_ids = batch['qa_input_ids'].to(device)
        # qa_target_ids = batch['qa_target_ids'].to(device)

        decoder_input_ids = torch.ones(B*4, 1, dtype=torch.long, device=device) * self.config.decoder_start_token_id

        objects_ids = batch['objects_ids']
        objects_ids_y = torch.zeros(len(objects_ids),10).to(device)
        for i in range(len(objects_ids)):
            kk = objects_ids[i].shape[1]
            if kk<=9:
                objects_ids_y[i][:kk] = objects_ids[i][:,:kk]
            else:
                objects_ids_y[i] = objects_ids[i][:,:10]


        output = self(
            input_ids=qa_input_ids,
            objects_ids=objects_ids_y,
            vis_inputs=(vis_feats, vis_pos),
            decoder_input_ids=decoder_input_ids,
            return_dict=True
        )

        logits = output['logits'].detach()
        logits = logits.view(B*4, self.lm_head.out_features)
        true_logit = logits[:, self.true_id].view(B*4, 1)
        false_logit = logits[:, self.false_id].view(B*4, 1)

        # [B*4, 2]
        binary_logits = torch.cat([true_logit, false_logit], dim=1)

        # [B*4]
        confidence = torch.softmax(binary_logits, dim=1)[:, 0]
        answer_id = confidence.view(B, 4).argmax(dim=1)

        result['qa_pred'] = answer_id.long().cpu().numpy()

        # QAR
        qar_input_ids = batch['qar_input_ids'].to(device)

        decoder_input_ids = torch.ones(B*4, 1, dtype=torch.long, device=device) * self.config.decoder_start_token_id

        output = self(
            input_ids=qar_input_ids,
            objects_ids=objects_ids_y,
            vis_inputs=(vis_feats, vis_pos),
            decoder_input_ids=decoder_input_ids,
            return_dict=True
        )

        logits = output['logits'].detach()
        logits = logits.view(B*4, self.lm_head.out_features)
        true_logit = logits[:, self.true_id].view(B*4, 1)
        false_logit = logits[:, self.false_id].view(B*4, 1)

        # [B*4, 2]
        binary_logits = torch.cat([true_logit, false_logit], dim=1)

        # [B*4]
        confidence = torch.softmax(binary_logits, dim=1)[:, 0]
        rationale_id = confidence.view(B, 4).argmax(dim=1)

        result['qar_pred'] = rationale_id.long().cpu().numpy()

        return result

    @torch.no_grad()
    def test_step(self, batch):
        self.eval()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        vis_pos = batch['boxes'].to(device)

        B = vis_pos.size(0)
        V_L = vis_pos.size(1)

        vis_feats = vis_feats.unsqueeze(1).expand(-1, 4, -1, -1).contiguous().view(B*4, V_L, 2048)
        vis_pos = vis_pos.unsqueeze(1).expand(-1, 4, -1, -1).contiguous().view(B*4, V_L, 4)

        result = {}

        # QA
        qa_input_ids = batch['qa_input_ids'].to(device)

        decoder_input_ids = torch.ones(B*4, 1, dtype=torch.long, device=device) * self.config.decoder_start_token_id

        objects_ids = batch['objects_ids']
        objects_ids_y = torch.zeros(len(objects_ids),10).to(device)
        for i in range(len(objects_ids)):
            kk = objects_ids[i].shape[1]
            if kk<=9:
                objects_ids_y[i][:kk] = objects_ids[i][:,:kk]
            else:
                objects_ids_y[i] = objects_ids[i][:,:10]


        output = self(
            input_ids=qa_input_ids,
            objects_ids=objects_ids_y,
            vis_inputs=(vis_feats, vis_pos),
            decoder_input_ids=decoder_input_ids,
            return_dict=True
        )

        logits = output['logits'].detach()
        logits = logits.view(B*4, self.lm_head.out_features)
        true_logit = logits[:, self.true_id].view(B*4, 1)
        false_logit = logits[:, self.false_id].view(B*4, 1)

        # [B*4, 2]
        binary_logits = torch.cat([true_logit, false_logit], dim=1)

        # [B*4]
        confidence = torch.softmax(binary_logits, dim=1)[:, 0]
        confidence = confidence.view(B, 4)
        answer_probs = confidence / confidence.sum(dim=1, keepdim=True)

        result['answer_probs'] = answer_probs.cpu().numpy()

        # QAR
        rationale_probs = torch.zeros(B, 4, 4)
        batch_qar_input_ids = batch['qar_input_ids'].view(B, 4, 4, -1)
        for i in range(4):
            qar_input_ids = batch_qar_input_ids[:, i].reshape(B*4, -1).to(device)

            decoder_input_ids = torch.ones(B*4, 1, dtype=torch.long, device=device) * self.config.decoder_start_token_id

            output = self(
                input_ids=qar_input_ids,
                objects_ids=objects_ids_y,
                vis_inputs=(vis_feats, vis_pos),
                decoder_input_ids=decoder_input_ids,
                return_dict=True
            )

            logits = output['logits'].detach().cpu()
            logits = logits.view(B*4, self.lm_head.out_features)
            true_logit = logits[:, self.true_id].view(B*4, 1)
            false_logit = logits[:, self.false_id].view(B*4, 1)

            # [B*4, 2]
            binary_logits = torch.cat([true_logit, false_logit], dim=1)

            # [B*16]
            confidence = torch.softmax(binary_logits, dim=1)[:, 0]
            confidence = confidence.view(B, 4)
            rationale_probs[:, i] = confidence / confidence.sum(dim=1, keepdim=True)

        result['rationale_probs'] = rationale_probs.numpy()
        return result

from modeling_bart import VLBart
class VLBartVCR(VLBart):
    def __init__(self, config):
        super().__init__(config)

        self.kl_div = nn.KLDivLoss(reduction='none')

    def train_step(self, batch):
        self.train()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        vis_pos = batch['boxes'].to(device)

        B = vis_pos.size(0)
        V_L = vis_pos.size(1)

        log_train_accuracy = batch['log_train_accuracy']

        result = {}

        # QA
        qa_input_ids = batch['qa_input_ids'].to(device)
        qa_target_ids = batch['qa_target_ids'].to(device)

        vis_feats = vis_feats.unsqueeze(1).expand(-1, 4, -1, -1).contiguous().view(B*4, V_L, 2048)
        vis_pos = vis_pos.unsqueeze(1).expand(-1, 4, -1, -1).contiguous().view(B*4, V_L, 4)

        output = self(
            input_ids=qa_input_ids,
            vis_inputs=(vis_feats, vis_pos),
            labels=qa_target_ids,
        )

        assert 'loss' in output

        qa_loss = output['loss'].view(B*4, -1)[:, 1].mean()

        if log_train_accuracy:
            logits = output['logits'].detach().view(B*4, -1, self.lm_head.out_features)[:, 1]
            logits = logits.view(B*4, self.lm_head.out_features)
            true_logit = logits[:, self.true_id].view(B*4, 1)
            false_logit = logits[:, self.false_id].view(B*4, 1)

            # [B*4, 2]
            binary_logits = torch.cat([true_logit, false_logit], dim=1)

            # [B*4]
            confidence = torch.softmax(binary_logits, dim=1)[:, 0]
            answer_id = confidence.view(B, 4).argmax(dim=1)

            result['qa_pred'] = answer_id.long().cpu().numpy()

        # QAR
        qar_input_ids = batch['qar_input_ids'].to(device)
        qar_target_ids = batch['qar_target_ids'].to(device)

        output = self(
            input_ids=qar_input_ids,
            vis_inputs=(vis_feats, vis_pos),
            labels=qar_target_ids,
        )
        assert 'loss' in output

        qar_loss = output['loss'].view(B*4, -1)[:, 1].mean()

        if log_train_accuracy:
            logits = output['logits'].detach().view(B*4, -1, self.lm_head.out_features)[:, 1]
            logits = logits.view(B*4, self.lm_head.out_features)
            true_logit = logits[:, self.true_id].view(B*4, 1)
            false_logit = logits[:, self.false_id].view(B*4, 1)

            # [B*4, 2]
            binary_logits = torch.cat([true_logit, false_logit], dim=1)

            # [B*4]
            confidence = torch.softmax(binary_logits, dim=1)[:, 0]
            rationale_id = confidence.view(B, 4).argmax(dim=1)

            result['qar_pred'] = rationale_id.long().cpu().numpy()

        loss = qa_loss + qar_loss

        result['loss'] = loss
        result['qa_loss'] = qa_loss.detach()
        result['qar_loss'] = qar_loss.detach()

        return result

    @torch.no_grad()
    def valid_step(self, batch):
        self.eval()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        vis_pos = batch['boxes'].to(device)

        B = vis_pos.size(0)
        V_L = vis_pos.size(1)

        result = {}

        vis_feats = vis_feats.unsqueeze(
            1).expand(-1, 4, -1, -1).contiguous().view(B*4, V_L, 2048)
        vis_pos = vis_pos.unsqueeze(
            1).expand(-1, 4, -1, -1).contiguous().view(B*4, V_L, 4)


        # QA
        qa_input_ids = batch['qa_input_ids'].to(device)

        decoder_input_ids = torch.tensor(
            [self.config.decoder_start_token_id, self.config.bos_token_id],
            dtype=torch.long, device=device).unsqueeze(0).expand(B*4, 2)

        output = self(
            input_ids=qa_input_ids,
            vis_inputs=(vis_feats, vis_pos),
            decoder_input_ids=decoder_input_ids,
            return_dict=True
        )

        logits = output['logits'].detach().view(B*4, -1, self.lm_head.out_features)[:, 1]
        logits = logits.view(B*4, self.lm_head.out_features)
        true_logit = logits[:, self.true_id].view(B*4, 1)
        false_logit = logits[:, self.false_id].view(B*4, 1)

        # [B*4, 2]
        binary_logits = torch.cat([true_logit, false_logit], dim=1)

        # [B*4]
        confidence = torch.softmax(binary_logits, dim=1)[:, 0]
        answer_id = confidence.view(B, 4).argmax(dim=1)

        result['qa_pred'] = answer_id.long().cpu().numpy()

        # QAR
        qar_input_ids = batch['qar_input_ids'].to(device)

        decoder_input_ids = torch.tensor(
            [self.config.decoder_start_token_id, self.config.bos_token_id],
            dtype=torch.long, device=device).unsqueeze(0).expand(B*4, 2)

        output = self(
            input_ids=qar_input_ids,
            vis_inputs=(vis_feats, vis_pos),
            decoder_input_ids=decoder_input_ids,
            return_dict=True
        )

        logits = output['logits'].detach().view(B*4, -1, self.lm_head.out_features)[:, 1]
        logits = logits.view(B*4, self.lm_head.out_features)
        true_logit = logits[:, self.true_id].view(B*4, 1)
        false_logit = logits[:, self.false_id].view(B*4, 1)

        # [B*4, 2]
        binary_logits = torch.cat([true_logit, false_logit], dim=1)

        # [B*4]
        confidence = torch.softmax(binary_logits, dim=1)[:, 0]
        rationale_id = confidence.view(B, 4).argmax(dim=1)

        result['qar_pred'] = rationale_id.long().cpu().numpy()

        return result

    @torch.no_grad()
    def test_step(self, batch):
        self.eval()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        vis_pos = batch['boxes'].to(device)

        B = vis_pos.size(0)
        V_L = vis_pos.size(1)

        vis_feats = vis_feats.unsqueeze(1).expand(-1, 4, -1, -1).contiguous().view(B*4, V_L, 2048)
        vis_pos = vis_pos.unsqueeze(1).expand(-1, 4, -1, -1).contiguous().view(B*4, V_L, 4)

        result = {}

        # QA
        qa_input_ids = batch['qa_input_ids'].to(device)

        decoder_input_ids = torch.tensor(
            [self.config.decoder_start_token_id, self.config.bos_token_id],
            dtype=torch.long, device=device).unsqueeze(0).expand(B*4, 2)

        output = self(
            input_ids=qa_input_ids,
            vis_inputs=(vis_feats, vis_pos),
            decoder_input_ids=decoder_input_ids,
            return_dict=True
        )

        logits = output['logits'].detach().view(B*4, -1, self.lm_head.out_features)[:, 1]
        logits = logits.view(B*4, self.lm_head.out_features)
        true_logit = logits[:, self.true_id].view(B*4, 1)
        false_logit = logits[:, self.false_id].view(B*4, 1)

        # [B*4, 2]
        binary_logits = torch.cat([true_logit, false_logit], dim=1)

        # [B*4]
        confidence = torch.softmax(binary_logits, dim=1)[:, 0]
        confidence = confidence.view(B, 4)
        answer_probs = confidence / confidence.sum(dim=1, keepdim=True)

        result['answer_probs'] = answer_probs.cpu().numpy()

        # QAR
        rationale_probs = torch.zeros(B, 4, 4)
        batch_qar_input_ids = batch['qar_input_ids'].view(B, 4, 4, -1)
        for i in range(4):
            qar_input_ids = batch_qar_input_ids[:, i].reshape(B*4, -1).to(device)

            decoder_input_ids = torch.tensor(
                [self.config.decoder_start_token_id, self.config.bos_token_id],
                dtype=torch.long, device=device).unsqueeze(0).expand(B*4, 2)

            output = self(
                input_ids=qar_input_ids,
                vis_inputs=(vis_feats, vis_pos),
                decoder_input_ids=decoder_input_ids,
                return_dict=True
            )

            logits = output['logits'].detach().view(B*4, -1, self.lm_head.out_features)[:, 1].cpu()
            logits = logits.view(B*4, self.lm_head.out_features)
            true_logit = logits[:, self.true_id].view(B*4, 1)
            false_logit = logits[:, self.false_id].view(B*4, 1)

            # [B*4, 2]
            binary_logits = torch.cat([true_logit, false_logit], dim=1)

            # [B*16]
            confidence = torch.softmax(binary_logits, dim=1)[:, 0]
            confidence = confidence.view(B, 4)
            rationale_probs[:, i] = confidence / confidence.sum(dim=1, keepdim=True)

        result['rationale_probs'] = rationale_probs.numpy()
        return result
