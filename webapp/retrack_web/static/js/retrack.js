// Shared page behaviour: Bootstrap form validation, confirmation modal, tooltips.
(() => {
  'use strict'

  // Bootstrap's client-side validation pattern. The server validates every request again (NFR-03).
  document.querySelectorAll('form.needs-validation').forEach(form => {
    form.addEventListener('submit', event => {
      if (!form.checkValidity()) {
        event.preventDefault()
        event.stopPropagation()
      }
      form.classList.add('was-validated')
    }, false)
  })

  // Buttons with data-confirm open the shared confirmation modal before their form is submitted.
  const modalEl = document.getElementById('confirmModal')
  if (modalEl) {
    const modal = new bootstrap.Modal(modalEl)
    let pending = null
    document.querySelectorAll('[data-confirm]').forEach(button => {
      button.addEventListener('click', event => {
        event.preventDefault()
        pending = button
        modalEl.querySelector('.modal-title').textContent = button.dataset.confirmTitle || 'Please confirm'
        modalEl.querySelector('.modal-body p').textContent = button.dataset.confirm
        const ok = modalEl.querySelector('[data-confirm-ok]')
        ok.className = 'btn ' + (button.dataset.confirmClass || 'btn-primary')
        ok.textContent = button.dataset.confirmLabel || button.textContent.trim()
        modal.show()
      })
    })
    modalEl.querySelector('[data-confirm-ok]').addEventListener('click', () => {
      if (pending) {
        const form = pending.form
        if (pending.name) {
          const hidden = document.createElement('input')
          hidden.type = 'hidden'
          hidden.name = pending.name
          hidden.value = pending.value
          form.appendChild(hidden)
        }
        modal.hide()
        form.submit()
      }
    })
  }

  document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => new bootstrap.Tooltip(el))
})()
