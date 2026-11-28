Extract the delivery note information from the following document.
The document may be in English, French, Spanish, or German.

Extract:
- delivery_note_number: unique identifier of the delivery note (BL / bon de livraison / Lieferscheinnummer / packing slip)
- order_number: customer or supplier order reference (PO number / commande / Bestellnummer) if present
- lines: list of product lines, each with product_number, quantity, lot_number, serial_number, and order_number.

Document:
{{ document }}
