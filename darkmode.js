document.addEventListener('DOMContentLoaded', function () {
  const toggle = document.getElementById('darkModeToggle');
  const body = document.body;

  function setDarkMode(on) {
    if (on) {
      body.classList.add('dark-mode');
      toggle.textContent = '☀️';
    } else {
      body.classList.remove('dark-mode');
      toggle.textContent = '🌙';
    }
  }

  toggle.addEventListener('click', () => {
    const isDark = !body.classList.contains('dark-mode');
    setDarkMode(isDark);
    localStorage.setItem('darkMode', isDark ? '1' : '0');
  });

  const darkPref = localStorage.getItem('darkMode');
  setDarkMode(darkPref === '1');
});
