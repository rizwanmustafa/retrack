/*!
 * Color mode toggler adapted from the Bootstrap 5.3 docs examples (https://getbootstrap.com/)
 * Copyright 2011-2024 The Bootstrap Authors. Licensed under the Creative Commons Attribution 3.0 Unported License.
 * Adapted to Bootstrap Icons font classes.
 */
(() => {
  'use strict'

  const getStoredTheme = () => { try { return localStorage.getItem('theme') } catch (e) { return null } }
  const setStoredTheme = theme => { try { localStorage.setItem('theme', theme) } catch (e) { /* ignored */ } }

  const getPreferredTheme = () => {
    const storedTheme = getStoredTheme()
    if (storedTheme) {
      return storedTheme
    }
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  }

  const setTheme = theme => {
    if (theme === 'auto') {
      document.documentElement.setAttribute('data-bs-theme',
        (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'))
    } else {
      document.documentElement.setAttribute('data-bs-theme', theme)
    }
  }

  setTheme(getPreferredTheme())

  const showActiveTheme = (theme, focus = false) => {
    const themeSwitcher = document.querySelector('#bd-theme')
    if (!themeSwitcher) {
      return
    }
    const activeThemeIcon = document.querySelector('.theme-icon-active')
    const btnToActive = document.querySelector(`[data-bs-theme-value="${theme}"]`)
    if (!btnToActive) {
      return
    }
    const iconOfActiveBtn = btnToActive.querySelector('i').dataset.icon

    document.querySelectorAll('[data-bs-theme-value]').forEach(element => {
      element.classList.remove('active')
      element.setAttribute('aria-pressed', 'false')
    })

    btnToActive.classList.add('active')
    btnToActive.setAttribute('aria-pressed', 'true')
    activeThemeIcon.className = `bi ${iconOfActiveBtn} theme-icon-active`
    themeSwitcher.setAttribute('aria-label', `Toggle theme (${btnToActive.dataset.bsThemeValue})`)

    if (focus) {
      themeSwitcher.focus()
    }
  }

  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    const storedTheme = getStoredTheme()
    if (storedTheme !== 'light' && storedTheme !== 'dark') {
      setTheme(getPreferredTheme())
    }
  })

  window.addEventListener('DOMContentLoaded', () => {
    showActiveTheme(getPreferredTheme())

    document.querySelectorAll('[data-bs-theme-value]')
      .forEach(toggle => {
        toggle.addEventListener('click', () => {
          const theme = toggle.getAttribute('data-bs-theme-value')
          setStoredTheme(theme)
          setTheme(theme)
          showActiveTheme(theme, true)
        })
      })
  })
})()
