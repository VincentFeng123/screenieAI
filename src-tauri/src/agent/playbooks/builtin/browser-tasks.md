---
name: browser-tasks
apps: com.apple.Safari, com.google.Chrome, company.thebrowser.Browser, com.microsoft.edgemac, com.brave.Browser
triggers: search, open, website, url, tab, browse, look up, price, buy, shop, compare
---
One-step actions first: openUrl for any URL or site you can name; webSearch for
general web queries. Never type into the address bar what openUrl or webSearch
can do in one step.

## Site search
For "search X on SITE" goals: openUrl the site first, wait for it to load, then
type the query into the site's own search field, not the address bar. If you
are certain of the site's search URL pattern, openUrl it directly with the
query embedded instead.

## Tabs
Reuse the current tab unless the user asked for a new one. If the address bar
already shows the URL or query you typed, press Return or wait; do not retype.

## Reading results
readPage returns the page text; compare results from it instead of clicking
into every result. Save prices or key facts with note before navigating away.
