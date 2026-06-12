use super::types::{
    CoordinateSpace, Element, ElementSource, ObservationError, PlatformElementHandle, Rect,
};
use serde::Deserialize;
use std::process::Command;

const DOM_EXTRACTION_JS: &str = r#"(function(){const ATTR='data-agent-id';const PREFIX='screenie-';const viewportWidth=window.innerWidth||document.documentElement.clientWidth||0;const viewportHeight=window.innerHeight||document.documentElement.clientHeight||0;const scrollX=window.scrollX||0;const scrollY=window.scrollY||0;const scroller=document.scrollingElement||document.documentElement;const scrollWidth=(scroller&&scroller.scrollWidth)||0;const scrollHeight=(scroller&&scroller.scrollHeight)||0;const used=new Set();let seq=0;function textOf(el){return (el.innerText||el.textContent||'').replace(/\s+/g,' ').trim();}function byIdList(value){return String(value||'').split(/\s+/).map(id=>document.getElementById(id)).filter(Boolean).map(textOf).filter(Boolean).join(' ');}function labelFor(el){const aria=el.getAttribute('aria-label');if(aria&&aria.trim())return aria.trim();const labelled=byIdList(el.getAttribute('aria-labelledby'));if(labelled)return labelled;const id=el.id;if(id){const label=document.querySelector('label[for="'+CSS.escape(id)+'"]');if(label){const text=textOf(label);if(text)return text;}}if(el.labels&&el.labels.length){const text=Array.from(el.labels).map(textOf).filter(Boolean).join(' ');if(text)return text;}for(const attr of ['title','placeholder','alt','name']){const value=el.getAttribute(attr);if(value&&value.trim())return value.trim();}const text=textOf(el);if(text)return text;return el.id||el.tagName.toLowerCase();}function inputType(el){return (el.getAttribute('type')||'text').toLowerCase();}function isEditable(el){return el.isContentEditable||el.getAttribute('contenteditable')===''||el.getAttribute('contenteditable')==='true';}function disabled(el){return !!el.disabled||String(el.getAttribute('aria-disabled')||'').toLowerCase()==='true';}function roleAttr(el){return (el.getAttribute('role')||'').toLowerCase();}function actionable(el){const tag=el.tagName.toLowerCase();const role=roleAttr(el);if(tag==='input')return inputType(el)!=='hidden';if(['button','select','textarea','summary'].includes(tag))return true;if(tag==='a'&&el.href)return true;if(isEditable(el))return true;if(['button','link','textbox','searchbox','combobox','checkbox','radio','slider','menuitem','menuitemradio','menuitemcheckbox','option','treeitem','tab','switch'].includes(role))return true;if(el.tabIndex>=0&&role)return true;return false;}function visibleStyle(el){const style=window.getComputedStyle(el);return style&&style.display!=='none'&&style.visibility!=='hidden'&&Number(style.opacity||'1')!==0;}function clipRect(rect){const left=Math.max(0,Math.min(viewportWidth,rect.left));const top=Math.max(0,Math.min(viewportHeight,rect.top));const right=Math.max(0,Math.min(viewportWidth,rect.right));const bottom=Math.max(0,Math.min(viewportHeight,rect.bottom));return {x:left,y:top,width:right-left,height:bottom-top};}function visibleRect(el){const rects=Array.from(el.getClientRects()).map(clipRect).filter(r=>r.width>=2&&r.height>=2);if(!rects.length)return null;const first=rects[0];let left=first.x,top=first.y,right=first.x+first.width,bottom=first.y+first.height;for(const r of rects.slice(1)){left=Math.min(left,r.x);top=Math.min(top,r.y);right=Math.max(right,r.x+r.width);bottom=Math.max(bottom,r.y+r.height);}return {x:left,y:top,width:right-left,height:bottom-top};}function unobscured(el,rect){const x=Math.max(0,Math.min(viewportWidth-1,rect.x+rect.width/2));const y=Math.max(0,Math.min(viewportHeight-1,rect.y+rect.height/2));const hit=document.elementFromPoint(x,y);return !hit||hit===el||el.contains(hit)||hit.contains(el);}function ensureId(el){let id=el.getAttribute(ATTR);if(id&&used.has(id))id='';if(!id){do{id=PREFIX+Date.now().toString(36)+'-'+(seq++).toString(36);}while(used.has(id));el.setAttribute(ATTR,id);}used.add(id);return id;}function labelProxyRect(el){if(el.tagName.toLowerCase()!=='input')return null;const t=inputType(el);if(t!=='radio'&&t!=='checkbox')return null;if(!el.labels||!el.labels.length)return null;for(const lab of Array.from(el.labels)){if(!visibleStyle(lab))continue;const r=visibleRect(lab);if(!r||r.width<2||r.height<2)continue;if(!unobscured(lab,r))continue;return r;}return null;}const selector='a[href],button,input,textarea,select,summary,[role],[tabindex],[contenteditable=""],[contenteditable="true"]';const elements=[];for(const el of Array.from(document.querySelectorAll(selector))){if(disabled(el)||!actionable(el))continue;let rect=null;if(visibleStyle(el)){rect=visibleRect(el);if(rect&&(rect.width<2||rect.height<2))rect=null;if(rect&&!unobscured(el,rect))rect=null;}if(!rect)rect=labelProxyRect(el);if(!rect)continue;const tag=el.tagName.toLowerCase();const type=tag==='input'?inputType(el):'';let value=null;if(tag==='input'&&(type==='radio'||type==='checkbox'))value=el.checked?'checked':'unchecked';else if((tag==='input'||tag==='textarea')&&type!=='password'&&typeof el.value==='string'&&el.value.trim())value=el.value;elements.push({agentId:ensureId(el),tag,inputType:type,role:roleAttr(el),name:labelFor(el),value,rect,disabled:disabled(el),focused:document.activeElement===el,contentEditable:isEditable(el),multiline:tag==='textarea'||el.getAttribute('aria-multiline')==='true'});}return JSON.stringify({innerWidth:viewportWidth,innerHeight:viewportHeight,scrollX:scrollX,scrollY:scrollY,scrollWidth:scrollWidth,scrollHeight:scrollHeight,elements});})()"#;

const DOM_REFRESH_JS_PREFIX: &str = r#"(function(agentId){const ATTR='data-agent-id';const viewportWidth=window.innerWidth||document.documentElement.clientWidth||0;const viewportHeight=window.innerHeight||document.documentElement.clientHeight||0;const scrollX=window.scrollX||0;const scrollY=window.scrollY||0;const scroller=document.scrollingElement||document.documentElement;const scrollWidth=(scroller&&scroller.scrollWidth)||0;const scrollHeight=(scroller&&scroller.scrollHeight)||0;function textOf(el){return (el.innerText||el.textContent||'').replace(/\s+/g,' ').trim();}function byIdList(value){return String(value||'').split(/\s+/).map(id=>document.getElementById(id)).filter(Boolean).map(textOf).filter(Boolean).join(' ');}function labelFor(el){const aria=el.getAttribute('aria-label');if(aria&&aria.trim())return aria.trim();const labelled=byIdList(el.getAttribute('aria-labelledby'));if(labelled)return labelled;const id=el.id;if(id){const label=document.querySelector('label[for="'+CSS.escape(id)+'"]');if(label){const text=textOf(label);if(text)return text;}}if(el.labels&&el.labels.length){const text=Array.from(el.labels).map(textOf).filter(Boolean).join(' ');if(text)return text;}for(const attr of ['title','placeholder','alt','name']){const value=el.getAttribute(attr);if(value&&value.trim())return value.trim();}const text=textOf(el);if(text)return text;return el.id||el.tagName.toLowerCase();}function inputType(el){return (el.getAttribute('type')||'text').toLowerCase();}function isEditable(el){return el.isContentEditable||el.getAttribute('contenteditable')===''||el.getAttribute('contenteditable')==='true';}function disabled(el){return !!el.disabled||String(el.getAttribute('aria-disabled')||'').toLowerCase()==='true';}function roleAttr(el){return (el.getAttribute('role')||'').toLowerCase();}function clipRect(rect){const left=Math.max(0,Math.min(viewportWidth,rect.left));const top=Math.max(0,Math.min(viewportHeight,rect.top));const right=Math.max(0,Math.min(viewportWidth,rect.right));const bottom=Math.max(0,Math.min(viewportHeight,rect.bottom));return {x:left,y:top,width:right-left,height:bottom-top};}function visibleRect(el){const rects=Array.from(el.getClientRects()).map(clipRect).filter(r=>r.width>=2&&r.height>=2);if(!rects.length)return null;const first=rects[0];let left=first.x,top=first.y,right=first.x+first.width,bottom=first.y+first.height;for(const r of rects.slice(1)){left=Math.min(left,r.x);top=Math.min(top,r.y);right=Math.max(right,r.x+r.width);bottom=Math.max(bottom,r.y+r.height);}return {x:left,y:top,width:right-left,height:bottom-top};}function visibleStyle(el){const style=window.getComputedStyle(el);return style&&style.display!=='none'&&style.visibility!=='hidden'&&Number(style.opacity||'1')!==0;}function unobscured(el,rect){const x=Math.max(0,Math.min(viewportWidth-1,rect.x+rect.width/2));const y=Math.max(0,Math.min(viewportHeight-1,rect.y+rect.height/2));const hit=document.elementFromPoint(x,y);return !hit||hit===el||el.contains(hit)||hit.contains(el);}function labelProxyRect(el){if(el.tagName.toLowerCase()!=='input')return null;const t=inputType(el);if(t!=='radio'&&t!=='checkbox')return null;if(!el.labels||!el.labels.length)return null;for(const lab of Array.from(el.labels)){if(!visibleStyle(lab))continue;const r=visibleRect(lab);if(!r||r.width<2||r.height<2)continue;if(!unobscured(lab,r))continue;return r;}return null;}const el=document.querySelector('['+ATTR+'="'+CSS.escape(agentId)+'"]');if(!el)return JSON.stringify({innerWidth:viewportWidth,innerHeight:viewportHeight,scrollX:scrollX,scrollY:scrollY,scrollWidth:scrollWidth,scrollHeight:scrollHeight,elements:[]});let rect=null;if(visibleStyle(el)){rect=visibleRect(el);if(rect&&(rect.width<2||rect.height<2))rect=null;if(rect&&!unobscured(el,rect))rect=null;}if(!rect)rect=labelProxyRect(el);if(!rect)return JSON.stringify({innerWidth:viewportWidth,innerHeight:viewportHeight,scrollX:scrollX,scrollY:scrollY,scrollWidth:scrollWidth,scrollHeight:scrollHeight,elements:[]});const tag=el.tagName.toLowerCase();const type=tag==='input'?inputType(el):'';let value=null;if(tag==='input'&&(type==='radio'||type==='checkbox'))value=el.checked?'checked':'unchecked';else if((tag==='input'||tag==='textarea')&&type!=='password'&&typeof el.value==='string'&&el.value.trim())value=el.value;return JSON.stringify({innerWidth:viewportWidth,innerHeight:viewportHeight,scrollX:scrollX,scrollY:scrollY,scrollWidth:scrollWidth,scrollHeight:scrollHeight,elements:[{agentId:agentId,tag,inputType:type,role:roleAttr(el),name:labelFor(el),value,rect,disabled:disabled(el),focused:document.activeElement===el,contentEditable:isEditable(el),multiline:tag==='textarea'||el.getAttribute('aria-multiline')==='true'}]});})("#;

const DOM_PRESS_JS_PREFIX: &str = r#"(function(agentId){const ATTR='data-agent-id';const viewportWidth=window.innerWidth||document.documentElement.clientWidth||0;const viewportHeight=window.innerHeight||document.documentElement.clientHeight||0;function disabled(el){return !!el.disabled||String(el.getAttribute('aria-disabled')||'').toLowerCase()==='true';}function visibleStyle(el){const style=window.getComputedStyle(el);return style&&style.display!=='none'&&style.visibility!=='hidden'&&Number(style.opacity||'1')!==0;}function clipRect(rect){const left=Math.max(0,Math.min(viewportWidth,rect.left));const top=Math.max(0,Math.min(viewportHeight,rect.top));const right=Math.max(0,Math.min(viewportWidth,rect.right));const bottom=Math.max(0,Math.min(viewportHeight,rect.bottom));return {x:left,y:top,width:right-left,height:bottom-top};}function visibleRect(el){const rects=Array.from(el.getClientRects()).map(clipRect).filter(r=>r.width>=2&&r.height>=2);if(!rects.length)return null;const first=rects[0];let left=first.x,top=first.y,right=first.x+first.width,bottom=first.y+first.height;for(const r of rects.slice(1)){left=Math.min(left,r.x);top=Math.min(top,r.y);right=Math.max(right,r.x+r.width);bottom=Math.max(bottom,r.y+r.height);}return {x:left,y:top,width:right-left,height:bottom-top};}function unobscured(el,rect){const x=Math.max(0,Math.min(viewportWidth-1,rect.x+rect.width/2));const y=Math.max(0,Math.min(viewportHeight-1,rect.y+rect.height/2));const hit=document.elementFromPoint(x,y);return !hit||hit===el||el.contains(hit)||hit.contains(el);}const el=document.querySelector('['+ATTR+'="'+CSS.escape(agentId)+'"]');if(!el)return JSON.stringify({pressed:false,reason:'missing'});if(disabled(el))return JSON.stringify({pressed:false,reason:'disabled'});if(!visibleStyle(el))return JSON.stringify({pressed:false,reason:'not visible'});const rect=visibleRect(el);if(!rect)return JSON.stringify({pressed:false,reason:'no visible rect'});if(!unobscured(el,rect))return JSON.stringify({pressed:false,reason:'center obscured'});try{if(typeof el.focus==='function')el.focus({preventScroll:true});}catch(e){try{el.focus();}catch(_){}}el.click();return JSON.stringify({pressed:true});})("#;

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SafariDomPayload {
    pub inner_width: f64,
    pub inner_height: f64,
    #[serde(default)]
    pub scroll_x: f64,
    #[serde(default)]
    pub scroll_y: f64,
    #[serde(default)]
    pub scroll_width: f64,
    #[serde(default)]
    pub scroll_height: f64,
    pub elements: Vec<SafariDomElement>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SafariDomElement {
    pub agent_id: String,
    pub tag: String,
    #[serde(default)]
    pub input_type: String,
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub value: Option<String>,
    pub rect: DomRect,
    #[serde(default)]
    pub disabled: bool,
    #[serde(default)]
    pub focused: bool,
    #[serde(default)]
    pub content_editable: bool,
    #[serde(default)]
    pub multiline: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SafariDomPressResult {
    pressed: bool,
    #[serde(default)]
    reason: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DomRect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

const PAGE_TEXT_JS: &str = r#"(function(){return (document.body&&document.body.innerText||'').replace(/\s+/g,' ').slice(0,6000);})()"#;

pub(crate) fn observe_safari_dom(
    web_area_bounds: Rect,
    start_id: u32,
) -> Result<Vec<Element>, ObservationError> {
    let payload = run_safari_dom_script(DOM_EXTRACTION_JS)?;
    Ok(elements_from_payload(payload, web_area_bounds, start_id))
}

/// Extract the visible text of the current Safari page for the agent's
/// readPage action. Reuses the osascript plumbing (and its Automation /
/// "Allow JavaScript from Apple Events" error guidance).
pub(crate) fn read_safari_page_text() -> Result<String, ObservationError> {
    let script = format!(
        "tell application \"Safari\" to do JavaScript {} in current tab of front window",
        applescript_string(PAGE_TEXT_JS)
    );
    let output = Command::new("/usr/bin/osascript")
        .arg("-e")
        .arg(script)
        .output()
        .map_err(|err| {
            ObservationError::SafariDomReadFailed(format!(
                "Safari page text read failed: could not run osascript: {err}"
            ))
        })?;

    if !output.status.success() {
        return Err(map_osascript_failure(
            output.status.code(),
            &String::from_utf8_lossy(&output.stderr),
        ));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

pub(crate) fn refresh_safari_dom_element(
    agent_id: &str,
    web_area_bounds: Rect,
    id: u32,
) -> Result<Option<Element>, ObservationError> {
    let js = format!(
        "{}{})",
        DOM_REFRESH_JS_PREFIX,
        serde_json::to_string(agent_id).map_err(|err| {
            ObservationError::SafariDomReadFailed(format!(
                "Safari DOM read failed: could not encode element id: {err}"
            ))
        })?
    );
    let payload = run_safari_dom_script(&js)?;
    Ok(elements_from_payload(payload, web_area_bounds, id)
        .into_iter()
        .next()
        .map(|mut element| {
            element.id = id;
            element
        }))
}

pub(crate) fn press_safari_dom_element(agent_id: &str) -> Result<bool, ObservationError> {
    let js = format!(
        "{}{})",
        DOM_PRESS_JS_PREFIX,
        serde_json::to_string(agent_id).map_err(|err| {
            ObservationError::SafariDomReadFailed(format!(
                "Safari DOM press failed: could not encode element id: {err}"
            ))
        })?
    );
    let raw = run_safari_javascript(&js, "Safari DOM press failed")?;
    let result: SafariDomPressResult = serde_json::from_str(raw.trim()).map_err(|err| {
        ObservationError::SafariDomReadFailed(format!(
            "Safari DOM press failed: invalid press JSON: {err}"
        ))
    })?;
    if !result.pressed {
        if let Some(reason) = result.reason {
            eprintln!("[screenie] agent safari-dom press declined: {reason}");
        }
    }
    Ok(result.pressed)
}

fn run_safari_dom_script(js: &str) -> Result<SafariDomPayload, ObservationError> {
    let stdout = run_safari_javascript(js, "Safari DOM read failed")?;
    parse_payload(stdout.trim())
}

fn run_safari_javascript(js: &str, failure_context: &str) -> Result<String, ObservationError> {
    let script = format!(
        "tell application \"Safari\" to do JavaScript {} in current tab of front window",
        applescript_string(js)
    );
    let output = Command::new("/usr/bin/osascript")
        .arg("-e")
        .arg(script)
        .output()
        .map_err(|err| {
            ObservationError::SafariDomReadFailed(format!(
                "{failure_context}: could not run osascript: {err}"
            ))
        })?;

    if !output.status.success() {
        return Err(map_osascript_failure(
            output.status.code(),
            &String::from_utf8_lossy(&output.stderr),
        ));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

pub(crate) fn parse_payload(json: &str) -> Result<SafariDomPayload, ObservationError> {
    serde_json::from_str(json).map_err(|err| {
        ObservationError::SafariDomReadFailed(format!(
            "Safari DOM read failed: invalid DOM JSON: {err}"
        ))
    })
}

pub(crate) fn map_osascript_failure(code: Option<i32>, stderr: &str) -> ObservationError {
    let lower = stderr.to_ascii_lowercase();
    if stderr.contains("-1743") || lower.contains("not authorized to send apple events") {
        return ObservationError::SafariDomReadFailed(
            "Safari Automation access is required. Allow this app to control Safari in System Settings > Privacy & Security > Automation, then try again."
                .into(),
        );
    }
    if lower.contains("javascript")
        && (lower.contains("not allowed")
            || lower.contains("disabled")
            || lower.contains("allow javascript from apple events"))
    {
        return ObservationError::SafariDomReadFailed(
            "Safari JavaScript from Apple Events is disabled. In Safari, enable Develop > Allow JavaScript from Apple Events, then try again."
                .into(),
        );
    }

    let detail = stderr.trim();
    let detail = if detail.is_empty() {
        format!("osascript exited with status {:?}", code)
    } else {
        format!("osascript exited with status {:?}: {detail}", code)
    };
    ObservationError::SafariDomReadFailed(format!("Safari DOM read failed: {detail}"))
}

pub(crate) fn elements_from_payload(
    payload: SafariDomPayload,
    web_area_bounds: Rect,
    start_id: u32,
) -> Vec<Element> {
    let geometry = SafariDomPayload {
        inner_width: payload.inner_width,
        inner_height: payload.inner_height,
        scroll_x: payload.scroll_x,
        scroll_y: payload.scroll_y,
        scroll_width: payload.scroll_width,
        scroll_height: payload.scroll_height,
        elements: Vec::new(),
    };

    payload
        .elements
        .into_iter()
        .filter(|item| !item.disabled)
        .filter_map(|item| dom_element_to_element(&geometry, item, web_area_bounds))
        .take(super::observer::MAX_OBSERVED_ELEMENTS)
        .enumerate()
        .map(|(index, mut element)| {
            element.id = start_id.saturating_add(index as u32);
            element.refresh_signature();
            element
        })
        .collect()
}

fn dom_element_to_element(
    payload: &SafariDomPayload,
    item: SafariDomElement,
    web_area_bounds: Rect,
) -> Option<Element> {
    let bounds = dom_rect_to_screen_rect(item.rect, web_area_bounds, payload)?;
    let role = normalize_dom_role(&item);
    let name = item.name.split_whitespace().collect::<Vec<_>>().join(" ");
    if role.is_empty() || name.is_empty() {
        return None;
    }

    Some(
        Element::new(
            0,
            role,
            name,
            item.value
                .map(|value| value.split_whitespace().collect::<Vec<_>>().join(" "))
                .filter(|value| !value.is_empty()),
            bounds,
            true,
            item.focused,
            CoordinateSpace::AxPoints,
            ElementSource::Web,
        )
        .with_platform_handle(PlatformElementHandle::SafariDom {
            agent_id: item.agent_id,
        }),
    )
}

pub(crate) fn dom_rect_to_screen_rect(
    rect: DomRect,
    web_area_bounds: Rect,
    payload: &SafariDomPayload,
) -> Option<Rect> {
    if !rect_is_finite(web_area_bounds)
        || !dom_rect_is_finite(rect)
        || payload.inner_width <= 0.0
        || !payload.inner_width.is_finite()
    {
        return None;
    }

    let scale = web_area_bounds.width / payload.inner_width;
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }

    let viewport_width = payload.inner_width.max(0.0);
    let viewport_height = if payload.inner_height.is_finite() && payload.inner_height > 0.0 {
        payload.inner_height
    } else {
        web_area_bounds.height / scale
    };
    let ax_height_css = web_area_bounds.height / scale;
    let (viewport_origin_x, viewport_origin_y) =
        if web_area_origin_is_scrolled(ax_height_css, payload) {
            (
                web_area_bounds.x + payload.scroll_x * scale,
                web_area_bounds.y + payload.scroll_y * scale,
            )
        } else {
            (web_area_bounds.x, web_area_bounds.y)
        };
    let clipped = clip_dom_rect(rect, viewport_width, viewport_height)?;
    Some(Rect {
        x: viewport_origin_x + clipped.x * scale,
        y: viewport_origin_y + clipped.y * scale,
        width: clipped.width * scale,
        height: clipped.height * scale,
    })
}

const AX_WEB_AREA_VIEWPORT_EPSILON_CSS_PX: f64 = 4.0;

/// WebKit usually exposes Safari's AXWebArea as a full-document rect whose
/// origin shifts up/left by the page scroll (origin.y ≈ chrome bottom −
/// scrollY), so the viewport's screen origin is web_area.origin + scroll ×
/// scale. Some configurations report a static viewport-sized rect instead;
/// distinguish them by which document metric the AX height tracks. At
/// scroll == 0 both branches agree, so misclassifying an unscrolled page is
/// harmless, and payloads without scroll fields default to zero offset
/// (legacy behavior).
fn web_area_origin_is_scrolled(ax_height_css: f64, payload: &SafariDomPayload) -> bool {
    if (ax_height_css - payload.inner_height).abs() <= AX_WEB_AREA_VIEWPORT_EPSILON_CSS_PX {
        return false;
    }
    payload.scroll_height > payload.inner_height
        && (ax_height_css - payload.scroll_height).abs()
            < (ax_height_css - payload.inner_height).abs()
}

fn clip_dom_rect(rect: DomRect, viewport_width: f64, viewport_height: f64) -> Option<DomRect> {
    let min_x = rect.x.max(0.0).min(viewport_width);
    let min_y = rect.y.max(0.0).min(viewport_height);
    let max_x = (rect.x + rect.width).max(0.0).min(viewport_width);
    let max_y = (rect.y + rect.height).max(0.0).min(viewport_height);
    let clipped = DomRect {
        x: min_x,
        y: min_y,
        width: max_x - min_x,
        height: max_y - min_y,
    };
    (clipped.width > 0.0 && clipped.height > 0.0).then_some(clipped)
}

fn normalize_dom_role(item: &SafariDomElement) -> String {
    // Password inputs unify on the AX secure role so the safety gate guards
    // them regardless of source (the DOM script already nulls their values).
    if item.tag == "input" && item.input_type == "password" {
        return "AXSecureTextField".into();
    }

    match item.role.as_str() {
        "button" | "tab" | "switch" => return "AXButton".into(),
        "link" => return "AXLink".into(),
        "textbox" | "searchbox" if item.multiline => return "AXTextArea".into(),
        "textbox" | "searchbox" => return "AXTextField".into(),
        "combobox" | "listbox" => return "AXComboBox".into(),
        "checkbox" => return "AXCheckBox".into(),
        "radio" => return "AXRadioButton".into(),
        "slider" => return "AXSlider".into(),
        // Custom ARIA dropdowns (Google Flights' cabin-class picker, etc.)
        // render their entries as `li role="option" tabindex="-1"` — without
        // a mapping the open dropdown's options vanish from the observation
        // while readPage and screenshots still show them, and clickText can
        // never resolve. AXMenuItem keeps them on the semantic-press path.
        "menuitem" | "menuitemradio" | "menuitemcheckbox" | "option" => {
            return "AXMenuItem".into()
        }
        "treeitem" => return "AXRow".into(),
        _ => {}
    }

    match item.tag.as_str() {
        "a" => "AXLink",
        "button" | "summary" => "AXButton",
        "select" => "AXPopUpButton",
        "textarea" => "AXTextArea",
        "input" => match item.input_type.as_str() {
            "button" | "submit" | "reset" | "image" | "file" => "AXButton",
            "checkbox" => "AXCheckBox",
            "radio" => "AXRadioButton",
            "range" => "AXSlider",
            _ => "AXTextField",
        },
        _ if item.content_editable => "AXTextArea",
        _ => "",
    }
    .into()
}

fn applescript_string(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for ch in value.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            _ => escaped.push(ch),
        }
    }
    escaped.push('"');
    escaped
}

fn rect_is_finite(rect: Rect) -> bool {
    rect.x.is_finite()
        && rect.y.is_finite()
        && rect.width.is_finite()
        && rect.height.is_finite()
        && rect.width > 0.0
        && rect.height > 0.0
}

fn dom_rect_is_finite(rect: DomRect) -> bool {
    rect.x.is_finite() && rect.y.is_finite() && rect.width.is_finite() && rect.height.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dom_rect_conversion_uses_width_scale_and_clips_to_viewport() {
        let payload = SafariDomPayload {
            inner_width: 800.0,
            inner_height: 600.0,
            scroll_x: 0.0,
            scroll_y: 0.0,
            scroll_width: 0.0,
            scroll_height: 0.0,
            elements: Vec::new(),
        };
        let web_area = Rect {
            x: 100.0,
            y: 200.0,
            width: 400.0,
            height: 300.0,
        };

        let converted = dom_rect_to_screen_rect(
            DomRect {
                x: 700.0,
                y: 580.0,
                width: 200.0,
                height: 40.0,
            },
            web_area,
            &payload,
        )
        .unwrap();

        assert_eq!(
            converted,
            Rect {
                x: 450.0,
                y: 490.0,
                width: 50.0,
                height: 10.0,
            }
        );
    }

    #[test]
    fn password_inputs_map_to_the_secure_text_role() {
        let secure = SafariDomElement {
            agent_id: "screenie-pw".into(),
            tag: "input".into(),
            input_type: "password".into(),
            role: "".into(),
            name: "Password".into(),
            value: None,
            rect: DomRect {
                x: 10.0,
                y: 10.0,
                width: 200.0,
                height: 30.0,
            },
            disabled: false,
            focused: false,
            content_editable: false,
            multiline: false,
        };
        assert_eq!(normalize_dom_role(&secure), "AXSecureTextField");

        // Even with a generic textbox role attribute, password wins.
        let with_role = SafariDomElement {
            role: "textbox".into(),
            ..secure
        };
        assert_eq!(normalize_dom_role(&with_role), "AXSecureTextField");
    }

    #[test]
    fn aria_listbox_options_map_to_clickable_roles() {
        // Field failure: Google Flights' cabin-class dropdown renders its
        // entries as `li role="option" tabindex="-1"`. Without a mapping
        // they were dropped at the empty-role check and the open dropdown
        // was invisible to clickText/findUi while readPage and screenshots
        // still showed it.
        let option = SafariDomElement {
            agent_id: "screenie-opt".into(),
            tag: "li".into(),
            input_type: "".into(),
            role: "option".into(),
            name: "First".into(),
            value: None,
            rect: DomRect {
                x: 10.0,
                y: 10.0,
                width: 100.0,
                height: 20.0,
            },
            disabled: false,
            focused: false,
            content_editable: false,
            multiline: false,
        };
        assert_eq!(normalize_dom_role(&option), "AXMenuItem");
        assert_eq!(
            normalize_dom_role(&SafariDomElement {
                role: "menuitemradio".into(),
                ..option.clone()
            }),
            "AXMenuItem"
        );
        assert_eq!(
            normalize_dom_role(&SafariDomElement {
                role: "menuitemcheckbox".into(),
                ..option.clone()
            }),
            "AXMenuItem"
        );
        assert_eq!(
            normalize_dom_role(&SafariDomElement {
                role: "treeitem".into(),
                ..option
            }),
            "AXRow"
        );
    }

    #[test]
    fn extraction_script_admits_aria_option_roles() {
        // The role mapping above is useless if the injected extractor's
        // actionable() whitelist rejects the nodes before they reach Rust.
        for role in ["'option'", "'menuitemradio'", "'menuitemcheckbox'", "'treeitem'"] {
            assert!(
                DOM_EXTRACTION_JS.contains(role),
                "actionable() whitelist is missing {role}"
            );
        }
    }

    #[test]
    fn payload_conversion_creates_web_elements_with_handles_and_ax_roles() {
        let payload = SafariDomPayload {
            inner_width: 1000.0,
            inner_height: 800.0,
            scroll_x: 0.0,
            scroll_y: 0.0,
            scroll_width: 0.0,
            scroll_height: 0.0,
            elements: vec![
                SafariDomElement {
                    agent_id: "screenie-a".into(),
                    tag: "input".into(),
                    input_type: "search".into(),
                    role: "".into(),
                    name: "Search products".into(),
                    value: Some("mac mini".into()),
                    rect: DomRect {
                        x: 100.0,
                        y: 50.0,
                        width: 300.0,
                        height: 40.0,
                    },
                    disabled: false,
                    focused: true,
                    content_editable: false,
                    multiline: false,
                },
                SafariDomElement {
                    agent_id: "screenie-b".into(),
                    tag: "a".into(),
                    input_type: "".into(),
                    role: "".into(),
                    name: "Details".into(),
                    value: None,
                    rect: DomRect {
                        x: 500.0,
                        y: 60.0,
                        width: 100.0,
                        height: 20.0,
                    },
                    disabled: false,
                    focused: false,
                    content_editable: false,
                    multiline: false,
                },
            ],
        };
        let elements = elements_from_payload(
            payload,
            Rect {
                x: 10.0,
                y: 20.0,
                width: 500.0,
                height: 400.0,
            },
            7,
        );

        assert_eq!(elements.len(), 2);
        assert_eq!(elements[0].id, 7);
        assert_eq!(elements[0].role, "AXTextField");
        assert_eq!(elements[0].source, ElementSource::Web);
        assert_eq!(elements[0].coordinate_space, CoordinateSpace::AxPoints);
        assert_eq!(
            elements[0].platform_handle,
            Some(PlatformElementHandle::SafariDom {
                agent_id: "screenie-a".into()
            })
        );
        assert_eq!(
            elements[0].bounds,
            Rect {
                x: 60.0,
                y: 45.0,
                width: 150.0,
                height: 20.0,
            }
        );
        assert_eq!(elements[1].id, 8);
        assert_eq!(elements[1].role, "AXLink");
    }

    #[test]
    fn scrolled_full_document_web_area_offsets_origin_by_scroll() {
        // The field repro: after the page scrolled 853 CSS px, WebKit's
        // full-document AXWebArea origin sat above the screen and the color
        // swatch (viewport y 753) was misplaced to screen y -10 instead of
        // its true on-screen position at y 843.
        let payload = SafariDomPayload {
            inner_width: 1440.0,
            inner_height: 900.0,
            scroll_x: 0.0,
            scroll_y: 853.0,
            scroll_width: 1440.0,
            scroll_height: 3000.0,
            elements: Vec::new(),
        };
        let web_area = Rect {
            x: 0.0,
            y: 90.0 - 853.0,
            width: 1440.0,
            height: 3000.0,
        };

        let converted = dom_rect_to_screen_rect(
            DomRect {
                x: 1103.0,
                y: 753.0,
                width: 36.0,
                height: 36.0,
            },
            web_area,
            &payload,
        )
        .unwrap();

        assert_eq!(
            converted,
            Rect {
                x: 1103.0,
                y: 843.0,
                width: 36.0,
                height: 36.0,
            }
        );
    }

    #[test]
    fn static_viewport_web_area_skips_scroll_offset() {
        // Guard for Safari builds whose AXWebArea is a viewport-sized rect:
        // its origin is already the viewport origin, so the scroll offset
        // must not be applied even when the page is scrolled.
        let payload = SafariDomPayload {
            inner_width: 1440.0,
            inner_height: 900.0,
            scroll_x: 0.0,
            scroll_y: 853.0,
            scroll_width: 1440.0,
            scroll_height: 3000.0,
            elements: Vec::new(),
        };
        let web_area = Rect {
            x: 0.0,
            y: 90.0,
            width: 1440.0,
            height: 900.0,
        };

        let converted = dom_rect_to_screen_rect(
            DomRect {
                x: 100.0,
                y: 200.0,
                width: 50.0,
                height: 20.0,
            },
            web_area,
            &payload,
        )
        .unwrap();

        assert_eq!(
            converted,
            Rect {
                x: 100.0,
                y: 290.0,
                width: 50.0,
                height: 20.0,
            }
        );
    }

    #[test]
    fn horizontally_scrolled_full_document_offsets_x() {
        let payload = SafariDomPayload {
            inner_width: 1000.0,
            inner_height: 800.0,
            scroll_x: 120.0,
            scroll_y: 0.0,
            scroll_width: 1600.0,
            scroll_height: 2400.0,
            elements: Vec::new(),
        };
        let web_area = Rect {
            x: -120.0,
            y: 80.0,
            width: 1000.0,
            height: 2400.0,
        };

        let converted = dom_rect_to_screen_rect(
            DomRect {
                x: 40.0,
                y: 10.0,
                width: 60.0,
                height: 20.0,
            },
            web_area,
            &payload,
        )
        .unwrap();

        assert_eq!(
            converted,
            Rect {
                x: 40.0,
                y: 90.0,
                width: 60.0,
                height: 20.0,
            }
        );
    }

    #[test]
    fn payload_without_scroll_fields_parses_with_zero_defaults() {
        let payload =
            parse_payload(r#"{"innerWidth":800,"innerHeight":600,"elements":[]}"#).unwrap();
        assert_eq!(payload.scroll_x, 0.0);
        assert_eq!(payload.scroll_y, 0.0);
        assert_eq!(payload.scroll_width, 0.0);
        assert_eq!(payload.scroll_height, 0.0);

        // Zero scroll metrics mean zero offset: conversion matches legacy.
        let converted = dom_rect_to_screen_rect(
            DomRect {
                x: 700.0,
                y: 580.0,
                width: 200.0,
                height: 40.0,
            },
            Rect {
                x: 100.0,
                y: 200.0,
                width: 400.0,
                height: 300.0,
            },
            &payload,
        )
        .unwrap();
        assert_eq!(
            converted,
            Rect {
                x: 450.0,
                y: 490.0,
                width: 50.0,
                height: 10.0,
            }
        );
    }

    #[test]
    fn elements_from_payload_applies_scroll_offset() {
        // Guards the stripped geometry clone inside elements_from_payload:
        // the scroll fields must survive into the per-element conversion.
        let payload = SafariDomPayload {
            inner_width: 1000.0,
            inner_height: 800.0,
            scroll_x: 0.0,
            scroll_y: 500.0,
            scroll_width: 1000.0,
            scroll_height: 4000.0,
            elements: vec![SafariDomElement {
                agent_id: "screenie-c".into(),
                tag: "a".into(),
                input_type: "".into(),
                role: "".into(),
                name: "Details".into(),
                value: None,
                rect: DomRect {
                    x: 10.0,
                    y: 100.0,
                    width: 100.0,
                    height: 50.0,
                },
                disabled: false,
                focused: false,
                content_editable: false,
                multiline: false,
            }],
        };
        let web_area = Rect {
            x: 0.0,
            y: 80.0 - 500.0,
            width: 1000.0,
            height: 4000.0,
        };

        let elements = elements_from_payload(payload, web_area, 1);
        assert_eq!(elements.len(), 1);
        assert_eq!(
            elements[0].bounds,
            Rect {
                x: 10.0,
                y: 180.0,
                width: 100.0,
                height: 50.0,
            }
        );
    }

    #[test]
    fn extraction_and_refresh_js_report_scroll_offsets() {
        for js in [DOM_EXTRACTION_JS, DOM_REFRESH_JS_PREFIX] {
            assert!(js.contains("scrollX:scrollX"), "scrollX missing");
            assert!(js.contains("scrollY:scrollY"), "scrollY missing");
            assert!(js.contains("scrollWidth:scrollWidth"), "scrollWidth missing");
            assert!(
                js.contains("scrollHeight:scrollHeight"),
                "scrollHeight missing"
            );
        }
        // Every refresh return branch (element missing, rect missing,
        // success) must carry the geometry so the payload parses uniformly.
        assert_eq!(DOM_REFRESH_JS_PREFIX.matches("scrollY:scrollY").count(), 3);
    }

    #[test]
    fn refresh_js_applies_extraction_occlusion_checks() {
        // The refresh script must reject hidden/occluded targets the same
        // way extraction does, so a sticky-header-covered element refreshes
        // to "no rect" and forces a replan instead of a click into the
        // header.
        for needle in ["function visibleStyle", "function unobscured"] {
            assert!(DOM_REFRESH_JS_PREFIX.contains(needle), "{needle} missing");
        }
        assert!(DOM_REFRESH_JS_PREFIX.contains("if(rect&&!unobscured(el,rect))rect=null;"));
        assert!(DOM_REFRESH_JS_PREFIX.contains("if(!unobscured(lab,r))continue;"));
    }

    #[test]
    fn press_js_checks_center_hit_before_clicking_element() {
        assert!(DOM_PRESS_JS_PREFIX.contains("document.elementFromPoint"));
        assert!(DOM_PRESS_JS_PREFIX.contains("unobscured(el,rect)"));
        assert!(DOM_PRESS_JS_PREFIX.contains("el.click()"));
    }

    #[test]
    fn extraction_js_surfaces_hidden_radio_inputs_through_their_labels() {
        // Apple's configurator pattern: visually-hidden radio inputs with
        // styled label tiles. The extractor must fall back to the label's
        // geometry instead of dropping the input (and the refresh script
        // must keep resolving it afterwards).
        for js in [DOM_EXTRACTION_JS, DOM_REFRESH_JS_PREFIX] {
            assert!(
                js.contains("labelProxyRect"),
                "label-proxy fallback missing from script"
            );
            assert!(
                js.contains("el.checked?'checked':'unchecked'"),
                "radio/checkbox state missing from script"
            );
        }
        // The fallback applies only to radio/checkbox inputs.
        assert!(DOM_EXTRACTION_JS.contains("t!=='radio'&&t!=='checkbox'"));
    }

    #[test]
    fn osascript_error_mapping_reports_actionable_safari_messages() {
        assert!(map_osascript_failure(
            Some(1),
            "execution error: Not authorized to send Apple events to Safari. (-1743)"
        )
        .to_string()
        .contains("Automation access"));
        assert!(map_osascript_failure(
            Some(1),
            "execution error: JavaScript execution is not allowed"
        )
        .to_string()
        .contains("Allow JavaScript from Apple Events"));
        assert!(map_osascript_failure(Some(1), "syntax error")
            .to_string()
            .contains("DOM read"));
    }
}
