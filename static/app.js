const examples = {
  refund: {
    subject: 'Refund for a duplicate charge',
    body: 'Hi team, I was charged twice for my Pro subscription this month. Could you refund the duplicate payment? The amount in question is $24.00. Thanks for your help.',
    customer_tier: 'pro', amount: 24,
    draft_reply: 'You can request a refund within 30 days of the charge. Approved refunds return to the original payment method within 5 business days.',
    kb_article: 'Refunds may be requested within 30 days of a charge. This applies to all plans. Approved refunds are returned to the original payment method and typically settle within 5 business days.'
  },
  account: {
    subject: 'Password reset link has expired',
    body: 'Hi, I cannot access my account. Every password reset link says it has expired, even when I click it immediately. I tried a different browser but still cannot log in.',
    customer_tier: 'free', amount: 0,
    draft_reply: 'Password reset links remain valid for 24 hours. Please request a new link from the sign-in page.',
    kb_article: 'Password reset links expire after 30 minutes. Request a fresh link from the sign-in page. If you receive multiple emails, use only the most recent link.'
  },
  uncertain: {
    subject: 'Several things to sort out',
    body: 'I cannot sign in, my order has not arrived, and I think my latest payment is wrong. Can you help me understand what happened and make an exception to your usual policy?',
    customer_tier: 'pro', amount: 175,
    draft_reply: 'We can refund the full payment immediately and deliver your order tomorrow.',
    kb_article: 'Contact support for account access issues. Shipping timelines vary by destination. Refund eligibility is reviewed individually and depends on the original payment date.'
  }
};
const form = document.getElementById('request-form');
const results = document.getElementById('results');
const runButton = document.getElementById('run-button');
const errorMessage = document.getElementById('request-error');
const fields = {
  subject: document.getElementById('subject'), body: document.getElementById('body'),
  customer_tier: document.getElementById('customer-tier'), amount: document.getElementById('amount'),
  draft_reply: document.getElementById('draft-reply'), kb_article: document.getElementById('kb-article')
};
let currentResponses = null;
let running = false;
let inputRevision = 0;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function setExample(name) {
  Object.entries(examples[name]).forEach(([key, value]) => { fields[key].value = value; });
  document.querySelectorAll('.example-button').forEach(button => {
    const active = button.dataset.example === name;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
  markEdited();
}

function markEdited() {
  inputRevision += 1;
  if (currentResponses) document.getElementById('results-status').textContent = 'INPUT CHANGED · RUN AGAIN';
}

function percentage(probability) {
  return `${(probability * 100).toFixed(1)}%`;
}

function makeBadge(response) {
  return element('span', `decision-badge${response.abstain ? ' review' : ''}`, response.abstain ? 'Abstained · human review' : 'Model recommendation');
}

function drawResult(decision, response) {
  const card = document.getElementById(`${decision}-card`);
  const output = card.querySelector('.card-output');
  output.replaceChildren();
  const prediction = decision === 'route' ? `${response.value.charAt(0).toUpperCase()}${response.value.slice(1)}` : decision === 'approve' ? (response.value ? 'Approve the action' : 'Hold for review') : (response.value ? 'Supported by the article' : 'Not fully supported');
  output.append(element('span', '', response.abstain && decision !== 'route' ? 'Human review needed' : prediction));
  const description = card.querySelector('.decision-description');
  if (decision === 'route') {
    description.textContent = response.abstain ? 'This queue is a tentative prediction. Ask a human to choose the destination.' : `${percentage(response.confidence)} probability for the recommended queue.`;
    const bars = card.querySelector('.route-bars');
    bars.replaceChildren();
    Object.entries(response.probs).sort((a, b) => b[1] - a[1]).forEach(([queue, probability]) => {
      const row = element('div', `queue-row${queue === response.value ? ' selected' : ''}`);
      row.append(element('span', '', queue.charAt(0).toUpperCase() + queue.slice(1)));
      const track = element('div', 'queue-track');
      track.setAttribute('role', 'meter');
      track.setAttribute('aria-label', `${queue} probability`);
      track.setAttribute('aria-valuemin', '0');
      track.setAttribute('aria-valuemax', '100');
      track.setAttribute('aria-valuenow', String(probability * 100));
      const fill = element('div', 'queue-fill');
      track.append(fill);
      row.append(track, element('span', 'queue-value', `${Math.round(probability * 100)}%`));
      bars.append(row);
      requestAnimationFrame(() => { fill.style.width = `${probability * 100}%`; });
    });
  } else {
    const confidence = response.value ? response.p_true : 1 - response.p_true;
    const ring = card.querySelector('.confidence-ring');
    ring.classList.remove('empty-ring');
    ring.classList.toggle('warning-ring', response.abstain || !response.value);
    ring.style.setProperty('--confidence', `${confidence * 100}%`);
    ring.setAttribute('aria-label', `${percentage(confidence)} probability for the displayed answer`);
    const number = element('span', '', percentage(confidence));
    number.append(element('small', '', 'CONFIDENCE'));
    ring.replaceChildren(number);
    description.textContent = `${response.abstain ? 'Tentative prediction. ' : ''}${percentage(response.p_true)} probability of ${decision === 'approve' ? 'approval' : 'being supported'}.${response.abstain ? ' Decision deferred to a human.' : ''}`;
  }
  const bottom = card.querySelector('.card-bottom');
  bottom.replaceChildren(makeBadge(response), element('span', 'card-latency', `${response.latency_ms.toFixed(2)} ms inference`));
  card.classList.remove('completed');
  void card.offsetWidth;
  card.classList.add('completed');
}

function validResponse(response, decision) {
  if (!response || response.decision !== decision || typeof response.abstain !== 'boolean' || !Number.isFinite(response.latency_ms) || response.latency_ms < 0) return false;
  if (decision === 'route') {
    const queues = ['billing', 'technical', 'account', 'shipping', 'general'];
    if (!queues.includes(response.value) || response.type !== 'choice' || !response.probs || !Number.isFinite(response.confidence)) return false;
    if (!queues.every(queue => Number.isFinite(response.probs[queue]) && response.probs[queue] >= 0 && response.probs[queue] <= 1)) return false;
    return Math.abs(queues.reduce((sum, queue) => sum + response.probs[queue], 0) - 1) <= 0.01 && Math.abs(response.confidence - response.probs[response.value]) <= 0.01;
  }
  return response.type === 'binary' && typeof response.value === 'boolean' && Number.isFinite(response.p_true) && response.p_true >= 0 && response.p_true <= 1;
}

async function runDecisions(event) {
  if (event) event.preventDefault();
  if (running || !form.reportValidity()) return;
  running = true;
  const revision = inputRevision;
  runButton.disabled = true;
  runButton.querySelector('.button-label').textContent = 'Considering the request…';
  results.setAttribute('aria-busy', 'true');
  document.getElementById('results-status').textContent = 'MODEL AT WORK';
  errorMessage.hidden = true;
  const state = Object.fromEntries(Object.entries(fields).map(([key, field]) => [key, key === 'amount' ? Number(field.value) : field.value]));
  const batchId = globalThis.crypto?.randomUUID?.() || `request-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const responses = {};
  try {
    for (const decision of ['route', 'approve', 'grounded']) {
      const requestId = `${batchId}-${decision}`;
      const response = await fetch('/v1/decide', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({request_id: requestId, decision, state}), signal: AbortSignal.timeout(20000)
      });
      if (!response.ok) throw new Error(`The service returned HTTP ${response.status}. Check that the model is loaded, then try again.`);
      const body = await response.json();
      if (!validResponse(body, decision) || body.request_id !== requestId) throw new Error('The service returned an unexpected response. Please check the server output.');
      responses[decision] = body;
    }
    currentResponses = responses;
    Object.entries(responses).forEach(([decision, response]) => drawResult(decision, response));
    const mean = Object.values(responses).reduce((sum, response) => sum + response.latency_ms, 0) / 3;
    const latency = document.getElementById('metric-latency');
    latency.replaceChildren(document.createTextNode(mean.toFixed(2)), element('small', '', 'ms'));
    document.getElementById('results-status').textContent = revision === inputRevision ? 'THREE DECISIONS, MADE' : 'INPUT CHANGED · RUN AGAIN';
    document.getElementById('raw-response').textContent = JSON.stringify(responses, null, 2);
    document.getElementById('raw-details').hidden = false;
    await updateHealth();
  } catch (error) {
    errorMessage.textContent = error.name === 'TimeoutError' || error.name === 'AbortError' ? 'The request took too long. Check the local service and try again.' : error instanceof TypeError ? 'Could not reach the local service. Start the server and try again.' : error.message;
    errorMessage.hidden = false;
    document.getElementById('results-status').textContent = currentResponses ? 'PREVIOUS RESULTS · RUN FAILED' : 'WAITING FOR THE SERVICE';
    await updateHealth();
  } finally {
    running = false;
    runButton.disabled = false;
    runButton.querySelector('.button-label').textContent = 'Make the decisions';
    results.setAttribute('aria-busy', 'false');
  }
}

async function updateHealth() {
  const status = document.getElementById('service-status');
  try {
    const response = await fetch('/health', {signal: AbortSignal.timeout(4000)});
    const health = await response.json();
    const online = response.ok && health.loaded === true;
    const modelName = document.getElementById('model-name');
    if (modelName) modelName.textContent = online && health.metadata?.name ? health.metadata.name : 'Three points of clarity';
    status.className = `service-status ${online ? 'online' : 'offline'}`;
    status.replaceChildren(element('span', 'status-dot'), document.createTextNode(online ? 'Model online' : 'Model unavailable'));
  } catch {
    status.className = 'service-status offline';
    status.replaceChildren(element('span', 'status-dot'), document.createTextNode('Service offline'));
  }
}

async function loadMetrics() {
  try {
    const response = await fetch('/v1/metrics', {signal: AbortSignal.timeout(4000)});
    if (!response.ok) return;
    const metrics = await response.json();
    if (!Number.isFinite(metrics.score)) return;
    const score = document.getElementById('metric-score');
    score.replaceChildren(document.createTextNode(metrics.score.toFixed(1)), element('small', '', '/ 100'));
    document.getElementById('metric-score-detail').textContent = Number.isFinite(metrics.n_records) ? `${metrics.n_records.toLocaleString()} dev records · local benchmark` : 'Local benchmark · not hidden-test results';
  } catch {}
}

document.querySelectorAll('.example-button').forEach(button => button.addEventListener('click', () => setExample(button.dataset.example)));
form.addEventListener('submit', runDecisions);
form.addEventListener('input', () => {
  document.querySelectorAll('.example-button').forEach(button => { button.classList.remove('active'); button.setAttribute('aria-pressed', 'false'); });
  markEdited();
});
document.addEventListener('keydown', event => {
  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') { event.preventDefault(); runDecisions(); }
});
document.getElementById('copy-response').addEventListener('click', async event => {
  const button = event.currentTarget;
  try {
    await navigator.clipboard.writeText(JSON.stringify(currentResponses, null, 2));
    button.textContent = 'Copied';
  } catch {
    button.textContent = 'Select JSON to copy';
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.getElementById('raw-response'));
    selection.removeAllRanges();
    selection.addRange(range);
  }
  setTimeout(() => { button.textContent = 'Copy JSON'; }, 1800);
});
setExample('refund');
updateHealth();
loadMetrics();
