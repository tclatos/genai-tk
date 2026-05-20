// BAML model for extracting structured data from delivery notes.
// Supports documents in English, French, or German.


class DeliveryNoteLine {

  product_number string @description("product or item reference number (EN: product number / part number, FR: numéro de produit / référence article, DE: Artikelnummer / Produktnummer / HTZ)")
  quantity float @description("quantity delivered (EN: quantity, FR: quantité, DE: Menge)")
  lot_number string? @description("manufacturing lot or batch number, if present (EN: lot/batch number, FR: numéro de lot, DE: Chargennummer)")
  serial_number string? @description("individual unit serial number, if present (EN: serial number, FR: numéro de série, DE: Seriennummer)")
  order_number string? @description("customer or supplier order reference, if present (EN: order number / PO number, FR: numéro de commande, DE: Bestellnummer)")
}


class DeliveryNote {
  delivery_note_number string @description("unique identifier of the delivery note (EN: delivery note number, FR: numéro de BL / bon de livraison, DE: Lieferscheinnummer, lieferung)")
  order_number string? @description("customer or supplier order reference, if present (EN: order number / PO number, FR: numéro de commande, DE: Bestellnummer)")
  lines DeliveryNoteLine[] @description("list of product lines on the delivery note")
}


// Extract a structured DeliveryNote from raw document text (Markdown / plain text).
function ExtractDeliveryNote(document: string) -> DeliveryNote {
  client "Gpt41mini"
  prompt #"
    Extract the delivery note information from the following markdown document, extracted from a PDF with OCR.
    The document may be in English, French, Spanish or German.

    Document:
    {{ document }}

    {{ ctx.output_format }}
  "#
}

// Extract a structured DeliveryNote directly from a PDF file.
function ExtractDeliveryNotePdf(document: pdf) -> DeliveryNote {
  client "Gpt41mini"
  prompt #"
    Extract the delivery note information from the attached PDF document.
    The document may be in English, French, Spanish or German.

    Here examples of delivery information to extract, depending of the provider: 
    -  Provider: DIEL
       product_number : 2127759200000
       order_number :  8210312328
    -  Provider: LAUAK
       product_number : V2121888600200
    -  Provider: MECACHROME
       product_number : V2121888600200
       delivery_note_number : BL 1548177
         

    {{ document }}

    {{ ctx.output_format }}
  "#
}


// Test with a French delivery note sample.
test french_delivery_note {
  functions [ExtractDeliveryNote]
  args {
    document #"
      BON DE LIVRAISON
      N° de BL : BL-2024-00123
      N° de commande : CMD-9876

      Lignes :
      | Réf. produit | Désignation          | N° de lot  | N° de série | Quantité |
      |--------------|----------------------|------------|-------------|----------|
      | ART-001      | Capteur de pression  | LOT-2024A  |             | 10       |
      | ART-002      | Valve de régulation  | LOT-2024B  | SN-00045    | 2        |
      | ART-003      | Filtre hydraulique   |            |             | 5        |
    "#
  }
}

// Test with a German delivery note sample.
test german_delivery_note {
  functions [ExtractDeliveryNote]
  args {
    document #"
      LIEFERSCHEIN
      Lieferschein-Nr.: LS-2024-00456
      Bestellnummer: BE-5432

      Positionen:
      | Artikelnummer | Bezeichnung        | Chargennr.  | Seriennr. | Menge |
      |---------------|--------------------|-------------|-----------|-------|
      | TL-100        | Drucksensor        | CH-2024-01  |           | 8     |
      | TL-200        | Regelventil        |             | SN-0078   | 1     |
    "#
  }
}

// Test with an English delivery note sample (no order number).
test english_delivery_note_no_order {
  functions [ExtractDeliveryNote]
  args {
    document #"
      DELIVERY NOTE
      Packing Slip #: DN-2024-00789

      Items:
      | Part Number | Description         | Batch No.  | Serial No. | Qty |
      |-------------|---------------------|------------|------------|-----|
      | P-500       | Hydraulic pump      | B-2024-03  |            | 3   |
      | P-501       | Pump coupling       |            | SN-00201   | 3   |
    "#
  }
}
